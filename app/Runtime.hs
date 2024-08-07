{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances #-}

{- |
The hMPC runtime module is used to execute secure multiparty computations.

Parties perform computations on secret-shared values by exchanging messages.
Shamir's threshold secret sharing scheme is used for finite fields of any order
exceeding the number of parties. hMPC provides many secure data types, ranging
from numeric types to more advanced types, for which the corresponding operations
are made available through Haskell's mechanism for operator overloading.
-}
module Runtime (secIntGen, secFldGen, runMpc, runMpcWithArgs, runSession, Input, input, Output(..), transfer, (.+), (.-), (.*), (./), srecip, (.^), (.<), (.<=), (.>), (.==), isZero, isZeroPublic, ssignum, argmaxfunc, argmax, smaximum, ssum, sproduct,sall, randomBits, inProd, schurProd, matrixProd, IfElse(..), ifElseList, async, await) where

import Control.Lens.Traversal
import Data.Maybe
import Data.List.Split
import Data.List
import Data.Bits
import Data.Function
import Text.Printf
import System.Info (os)
import Control.Concurrent
import Control.Monad
import Control.Monad.State
import Asyncoro
import System.Process
import System.Environment
import System.Random
import Shamir
import Prelude
import Types
import Network.Socket
import Parser
import SecTypes
import FinFields
import Data.Serialize (encode, decode, Serialize)
import qualified Data.ByteString as BS
import Data.Time
import System.Log.Logger
import System.Log.Formatter
import System.Log.Handler.Simple
import System.Log.Handler (setFormatter)
import System.IO
import Options.Applicative (Parser)

-- | Runs 'MPC' computation
runMpc :: SIO a -> IO a
runMpc = \action -> do
    conf <- Runtime.setup =<< Parser.getArgParser
    runSIO action conf

-- | Runs 'MPC' computation with user arguments
runMpcWithArgs :: Parser b -> (b -> SIO a) -> IO a
runMpcWithArgs parser = \action -> do
    (mpcOpts, userOpts) <- Parser.getArgParserExtra parser
    conf <- Runtime.setup mpcOpts
    runSIO (action userOpts) conf

-- | Start and Stop hMPC runtime
runSession :: SIO a -> SIO a
runSession action = do
    env <- Runtime.start
    liftIO $ runSIO (do val <- action; Runtime.shutdown; return val) env

exchangeShares :: [BS.ByteString] -> SIO [MVar BS.ByteString]
exchangeShares inShares = do
    parties <- gets parties

    forM_ (zip parties inShares) $ \(party, bytes) ->
        when (isJust $ sock party) (sendMessage bytes party)

    forM (zip parties [0..]) $ \(party, index) ->
        case sock party of
        Just _ -> receiveMessage party
        Nothing -> liftIO $ newMVar (inShares !! index)


-- | Transfer serializable Haskell objects
transfer :: (Serialize a) => a -> SIO (MVar [a])
transfer val = do
    parties <- gets parties
    let encVal = (encode val)  
    out <- async $ do
        forM_ parties $ \party -> 
            when (isJust $ sock party) (sendMessage encVal party)
  
        forM parties $ \party ->
            case sock party of
                Just _ -> do
                    bytes <- await =<< receiveMessage party
                    case (decode bytes) of
                        Right msg -> return msg
                Nothing -> return val
    return out


-- | Input x to the computation.
-- 
-- Value x is a secure object, or a list of secure objects.
class Input a b | a -> b where
  input :: a -> SIO b

instance Input (SIO SecureTypes) [SIO SecureTypes] where
    input a = map head <$> input [a]

instance Input [SIO SecureTypes] [[SIO SecureTypes]] where
    input xm = do
        x <- sequence xm
        Env{parties=parties, options=opts, gen=_gen} <- get

        outslist <- asyncListList (fromInteger (m opts)) (length x) $ do
            xf <- gather x
            let ftype = (head xf)
                length = (byteLength . meta) ftype
                (inShares, g'') = Shamir.randomSplit ftype xf (threshold opts) (m opts) _gen
            modify (\env -> env{gen = g''})

            forM_ (zip parties inShares) $ \(party, val) ->
                when (isJust $ sock party) (sendMessage (toBytes length val) party)
            
            forM (zip parties [0..]) $ \(party, index) -> do
                shares <- case sock party of
                    Just _ -> fromBytes length <$> (await =<< receiveMessage party)
                    Nothing -> return (inShares !! index)
                return $ map (\share -> ftype{value = share}) shares
        
        return $ (map . map) (\out -> return (head x){share = out}) outslist

class Reshare a b | a -> b where
  reshare :: a -> SIO b

instance Reshare FiniteField FiniteField where
    reshare a = head <$> reshare [a]

instance Reshare [FiniteField] [FiniteField] where
    reshare x = do
        Env{parties=parties, options=opts, gen=_gen} <- incPC >> get
        let ftype = head x
            length = (byteLength . meta) ftype
            (s, g'') = Shamir.randomSplit ftype x (threshold opts) (m opts) _gen
            inShares = map (toBytes length) s
        modify (\env -> env{gen = g''})
        shares <- exchangeShares inShares
        points <- forM (zip shares parties) $ \(share, party) -> do
                    val <- fromBytes length <$> (await share)
                    return ((pid party) + 1, val)

        return (Shamir.recombine ftype points)


-- | Output the value of x to the receivers specified.
-- Value x is a secure object, or a list of secure objects.
--
-- A secure integer is output as a Haskell Integer
class Output a b | a -> b where
  output :: a -> SIO (MVar b)

instance Output (SIO SecureTypes) Integer where
    output a = _output [a] head

instance Output [SIO SecureTypes] [Integer] where
    output a = _output a id

_output :: [SIO SecureTypes] -> ([Integer] -> b) -> SIO (MVar b)
_output xm convert = do
    x <- sequence xm
    parties <- gets parties
    out <- async $ do
        s <- gather x
        let inShares = map value s
            length = (byteLength . meta . head) s
        let inSharesEncoded = toBytes length inShares
        forM_ parties $ \party -> 
            when (isJust $ sock party) (sendMessage inSharesEncoded party)

        points <- forM parties $ \party -> do
            shares <- case sock party of
                Just _ -> fromBytes length <$> (await =<< receiveMessage party)
                Nothing -> return inShares
            return ((pid party) + 1, shares)
        
        let y = map value (recombine ((field . head) x) points)
        return $ convert y
    return out
    
    
shutdown :: SIO ()
shutdown = do
    Env{parties=parties, options=opt, forkIOBarrier=barrier, startTime=_startTime} <- get

    -- wait until all forkIO tasks have completed
    liftIO $ Asyncoro.decreaseBarrier barrier
    liftIO $ takeMVar (signal barrier)

    endTime <- liftIO $ getCurrentTime
    bytes <- mapM (await . nbytesSent) parties
    let elapsedTime = diffUTCTime endTime _startTime
    liftIO $ logging INFO $ printf "Computation time: %s sec | bytes sent: %d \n" (show elapsedTime) (sum bytes)    

    -- Synchronize with all parties before shutdown
    await =<< transfer (myPid opt)

    -- close connections peer_pid > pid
    liftIO $ forM_ (filter (\x -> ((pid x) > (myPid opt))) parties) $ \party -> do
        case (sock party) of
            Just _sock -> close _sock

-- | Secure addition of a and b.
(.+) :: SIO SecureTypes -> SIO SecureTypes -> SIO SecureTypes
(.+) am bm = do
    (a, b) <- sequenceOf both (am, bm)
    out <- async $ do 
        (af, bf) <- gather (a, b)
        return (af + bf)
    return (_coerce a b){share = out}

-- | Secure subtraction of a and b.
(.-) :: SIO SecureTypes -> SIO SecureTypes -> SIO SecureTypes
(.-) am bm = do
    (a, b) <- sequenceOf both (am, bm)
    out <- async $ do
        (af, bf) <- gather (a, b)
        return (af - bf)
    return (_coerce a b){share = out}    

-- | Secure multiplication of a and b.
(.*) :: SIO SecureTypes -> SIO SecureTypes -> SIO SecureTypes
(.*) am bm = do
    (a, b) <- sequenceOf both (am, bm)
    out <- async $ do
        (af, bf) <- gather (a, b)
        reshare (af * bf)
    return (_coerce a b){share = out}

-- | Secure division of a by b, for nonzero b.
(./) :: SIO SecureTypes -> SIO SecureTypes -> SIO SecureTypes
(./) am bm = do
    (recip bm) * am

-- | Secure reciprocal (multiplicative field inverse) of a, for nonzero a.
srecip :: SIO SecureTypes -> SIO SecureTypes
srecip am = do
    a <- am
    out <- async $ fix $ \loop -> do
            [r] <- _randoms a 1 Nothing
            ar <- await =<< output ((return a) .* (return r))
            if (ar == 0) 
                then loop
                else do 
                    rfld <- gather r
                    return (rfld / rfld{value = ar})
    return a{share = out}

-- | Secure exponentiation a raised to the power of b, for public integer b.
(.^) :: SIO SecureTypes -> Integer -> SIO SecureTypes
(.^) am b = do
    a <- am
    sproduct $ replicate (fromIntegral b) (return a)

-- | Secure comparison a < b.
(.<) :: SIO SecureTypes -> SIO SecureTypes -> SIO SecureTypes
(.<) am bm = ssignum True False (am - bm)

-- | Secure comparison a <= b.
(.<=) :: SIO SecureTypes -> SIO SecureTypes -> SIO SecureTypes
(.<=) am bm = 1 - (bm .< am)

-- | Secure comparison a > b.
(.>) :: SIO SecureTypes -> SIO SecureTypes -> SIO SecureTypes
(.>) am bm = (bm .< am)

-- | Secure comparison a == b.
(.==) :: SIO SecureTypes -> SIO SecureTypes -> SIO SecureTypes
(.==) am bm = isZero (am - bm)

-- | Secure zero test a == 0.
isZero :: SIO SecureTypes -> SIO SecureTypes
isZero am = do
    a <- am
    case a of
        SecFld {field=fld} -> (1 - ((return a) .^ ((modulus . meta) fld - 1))) -- todo modulus = order
        _ -> ssignum False True (return a)

-- | Secure public zero test of a.
isZeroPublic :: SIO SecureTypes -> SIO (MVar Bool)
isZeroPublic am = do
    a <- am
    [r] <- _randoms a 1 Nothing
    out <- async $ do
        (afld, rfld) <- gather (a, r)
        res <- await =<< (output $ setShare a (value (afld * rfld)))
        return (res == 0)
    return out

-- | Secure sign(um) of a, return -1 if a < 0 else 0 if a == 0 else 1.
--
-- If Boolean flag LT is set, perform a secure less than zero test instead, and
-- return 1 if a < 0 else 0, saving the work for a secure equality test.
-- If Boolean flag EQ is set, perform a secure equal to zero test instead, and
-- return 1 if a == 0 else 0, saving the work for a secure comparison.
ssignum :: Bool -> Bool -> SIO SecureTypes -> SIO SecureTypes
ssignum True True _ = error "lt and eq both true"
ssignum lt eq am = do
    a <- am
    opt <- gets options
    let l = (bitLength a)
    r_bits <- sequence =<< randomBits (return a) l False
    [r] <- _randoms a 1 $ Just $ 1 `shiftL` (secParam opt)
    out <- async $ do
        r_bits_fld <- gather r_bits
        let r_modl = foldr (\x acc -> (acc `shiftL` 1) + (value x)) 0 r_bits_fld

        (r_divl, af) <- gather (r, a)
        let a_rmodl = af + fromInteger ((1 `shiftL` l) + r_modl)
        c <- (`mod` (1 `shiftL` l)) 
                <$> (await =<< output (setShare a (value $ a_rmodl + fromInteger ((value r_divl) `shiftL` l))))

        z1 <- if not eq then do
            s_sign <- value <$> (randomBits (return a) 1 True >>= head >>= gather)
        
            let (e, sumXors) = foldl (\(e, sumXors) (bit, i) ->
                    let c_i = ((c `shiftR` i) .&. 1)
                    in (setShare a (s_sign + (value bit) - c_i + 3 * sumXors) : e,
                        sumXors + if c_i == 1 then 1 - (value bit) else (value bit)))
                    ([], 0) (zip (reverse r_bits_fld) [l-1, l-2..])

            g <- await =<< (isZeroPublic $ sproduct (setShare a (s_sign - 1 + 3*sumXors) : e))
            let h = if g then 3 - s_sign else 3 + s_sign
            return $ (fromInteger (c + (h `shiftL` (l - 1))) - a_rmodl) / fromInteger (1 `shiftL` l)
        else return FiniteField{}
        
        if not lt then do
            h <- await . share =<< sall (map (\(bit, i) -> 
                setShare a $ value $ if ((c `shiftR` i) .&. 1) == 1 then bit else 1 - bit) (zip r_bits_fld [0..])) 
            if eq then return h
            else reshare ((h - 1) * (2*z1 - 1))
        else return z1

    return a{share = out}

argmaxfunc :: [[SIO SecureTypes]] -> ([SIO SecureTypes] -> [SIO SecureTypes] -> SIO SecureTypes) -> SIO (SIO SecureTypes, [SIO SecureTypes])
argmaxfunc [xm] _ = do
    x <- sequence xm    
    return (setShare (head x) 0, map return x)
argmaxfunc x f = do
            let n = length x
            let (x0, x1) = splitAt (n `div` 2) x
            (i0, m0) <- argmaxfunc x0 f
            (i1, m1) <- argmaxfunc x1 f
            c <- return <$> f m0 m1
            a <- ifElse c (i1 + fromIntegral (n `div` 2)) i0
            m <- ifElse c m1 m0
            return (return a, m)


-- | Secure argmax of all given elements in x.
--
-- In case of multiple occurrences of the maximum values,
-- the index of the first occurrence is returned.
argmax :: [SIO SecureTypes] -> SIO (SIO SecureTypes, SIO SecureTypes)
argmax [xm] = do
    x <- xm
    return (setShare x 0, return x)
argmax x = do
    let n = length x
    let (x0, x1) = splitAt (n `div` 2) x 
    (i0, m0) <- argmax x0
    (i1, m1) <- argmax x1
    c <- return <$> (m0) .< (m1)
    a <- ifElse c (i1 + fromIntegral (n `div` 2)) i0
    m <- ifElse c m1 m0
    return (return a, return m)

-- | Secure maximum of all given elements in x, similar to Haskell's built-in maximum.
smaximum :: [SIO SecureTypes] -> SIO SecureTypes
smaximum [a] = a
smaximum x = do 
    let (x0, x1) = splitAt (length x `div` 2) x 
    m0 <- return <$> smaximum x0
    m1 <- return <$> smaximum x1
    (m0 .< m1) * (m1 - m0) + m0

-- | Secure sum of all elements in x, similar to Haskell's built-in sum.
ssum :: [SIO SecureTypes] -> SIO SecureTypes
ssum xm = do
    x <- sequence xm
    out <- async $ sum <$> gather x
    return (head x){share = out}

-- | Secure product of all elements in x, similar to Haskell's product.
--
-- Runs in log_2 len(x) rounds).
sproduct :: [SIO SecureTypes] -> SIO SecureTypes
sproduct xm = do
    x <- sequence xm
    out <- async $ head <$> iterate (\xold -> do
        (xmul, leftover) <- pairwise (*) <$> xold
        (leftover ++) <$> reshare xmul) (gather x) !! ((ceiling . logBase 2 . fromIntegral) (length x))        
    return (head x){share = out}

-- | Secure all of elements in x, similar to Haskell's built-in all.
--
-- Elements of x are assumed to be either 0 or 1 (Boolean).
-- Runs in log_2 len(x) rounds).
sall :: [SIO SecureTypes] -> SIO SecureTypes
sall xm = do
    x <- sequence xm
    out <- async $ head <$> iterate (\xold -> do
        (xmul, leftover) <- pairwise (*) <$> xold
        (leftover ++) <$> reshare xmul) (gather x) !! ((ceiling . logBase 2 . fromIntegral) (length x))        
    return (head x){share = out}

-- | Return n secure random values of the given type in the given range.
_randoms :: SecureTypes -> Integer -> Maybe Integer -> SIO [SecureTypes]
_randoms st n bound = do
    t <- threshold <$> (gets options)
    (g', g'') <- System.Random.split <$> gets gen
    let fld = field st
        _bound = case bound of
                Just b -> 1 `shiftL` max 0 ((floor . logBase 2 . fromIntegral) (b `div` (t + 1)))
                Nothing -> (modulus . meta) fld -- todo modulus = order
        x = take (fromIntegral n) $ randomRs (0, _bound - 1) g'
    modify (\env -> env{gen = g''})
    xlist <- input $ map (\rand -> do
                randmvar <- liftIO $ newMVar fld{value = rand} :: SIO (MVar FiniteField)
                return st{share = randmvar}) x
    forM (transpose xlist) $ \_x -> ssum _x

-- | Return n secure uniformly random bits of the given type.
randomBits :: SIO SecureTypes -> Int -> Bool -> SIO [SIO SecureTypes]
randomBits stm n signed = do
    st <- stm
    (g', g'') <- System.Random.split <$> gets gen
    let x = take (fromIntegral n) $ randomRs (0, 1) g'
    modify (\env -> env{gen = g''})
    xlist <- input $ map (\bit -> do
            randmvar <- liftIO $ newMVar (field st){value = ((2*bit)-1)} :: SIO (MVar FiniteField)
            return st{share = randmvar}) x

    secbits <- forM (transpose xlist) $ \x -> do
        let secbit = sproduct x   
        if signed then secbit
            else (secbit + 1) * fromInteger (((modulus . meta . field) st + 1) `shiftR` 1) -- todo modulus = characteristics
    
    return $ map (\secbit -> return secbit) secbits

-- | Secure dot product of x and y (one resharing).
inProd :: [SIO SecureTypes] -> [SIO SecureTypes] -> SIO SecureTypes
inProd xm ym = do
    (x, y) <- sequenceOf (both . traversed) (xm, ym)
    out <- async $ do
        (xf, yf) <- gather (x, y)
        reshare $ sum $ zipWith (*) xf yf
    return (head x){share = out}

-- | Secure entrywise multiplication of vectors x and y.
schurProd :: [SIO SecureTypes] -> [SIO SecureTypes] -> SIO [SIO SecureTypes]
schurProd xm ym = do
    (x, y) <- sequenceOf (both . traversed) (xm, ym)
    outs <- asyncList (length x) $ do
        (xf, yf) <- gather (x, y)
        reshare $ zipWith (*) xf yf
    return $ map (\out -> return (head x){share = out}) outs

-- | Secure matrix product of A with (transposed) B.
matrixProd :: [[SIO SecureTypes]] -> [[SIO SecureTypes]] -> Bool -> SIO [[SIO SecureTypes]]
matrixProd am bm tr = do
    (a, b) <- sequenceOf (both . traversed . traversed) (am, bm)
    let n2 = if tr then (length b) else (length (head b))
    outslist <- asyncListList (length a) n2 $ do
        (af, bf) <- gather (a, b)
        let bft = if tr then bf else transpose bf
        chunksOf n2 <$> reshare (concat [[sum $ zipWith (*) ai bi | bi <- bft] | ai <- af])

    return $ (map . map) (\out -> return ((head . head) a){share = out}) outslist

-- | Secure selection between x and y based on condition c.
class IfElse a b | a -> b where
    ifElse :: SIO SecureTypes -> a -> a -> SIO b

instance IfElse (SIO SecureTypes) (SecureTypes) where
    ifElse cm xm ym = do
        y <- return <$> ym
        cm * (xm - y) + y

instance IfElse [SIO SecureTypes] [SIO SecureTypes] where
    ifElse = ifElseList

ifElseList :: SIO SecureTypes -> [SIO SecureTypes] -> [SIO SecureTypes] -> SIO [SIO SecureTypes]
ifElseList am xm ym = do
    (x, y) <- sequenceOf (both . traversed) (xm, ym)
    a <- am
    outs <- asyncList (length x) $ do
        (af, xf, yf) <- gather (a, x, y)
        reshare $ map (\(x_i, y_i) -> x_i{value = (value af) * ((value x_i) - (value y_i)) + (value y_i)}) (zip xf yf)

    return $ map (\out -> return (head x){share = out}) outs
    
randomR' :: (Integer, Integer) -> SIO Integer
randomR' range = do
    _gen <- gets gen
    let (value, newGen) = randomR range _gen
    modify (\env -> env{gen = newGen})
    return value

pairwise :: (a -> a -> a) -> [a] -> ([a], [a])
pairwise _ [] = ([],[]) 
pairwise _ [x] = ([],[x])                           
pairwise f (x:y:xs) =
    let (z, leftover) = pairwise f xs
    in (f x y : z, leftover)

receiveMessage :: Party -> SIO (MVar BS.ByteString)
receiveMessage party = do
    pc <- gets pc 
    Asyncoro.receive pc party

-- | Send data to given peer, labeled by current program counter.
sendMessage :: BS.ByteString -> Party -> SIO ()
sendMessage bytes party = do
    pc <- gets pc
    Asyncoro.send pc bytes party
    
-- Start the hMPC runtime.
start :: SIO Env
start = do
    Env{parties=parties, options=opt} <- get
    liftIO $ do
        parties <- Asyncoro.createConnections (fromInteger $ myPid opt) parties
        countVar <- newMVar 1
        gen <- newStdGen
        signalVar <- newEmptyMVar
        startTime <- getCurrentTime
        return $ Env parties 0 opt (Barrier countVar signalVar) gen startTime

setup :: Options -> IO Env
setup opt = do
    setNumCapabilities =<< maybe getNumCapabilities return (nrThreads opt)
    h <- streamHandler stdout INFO >>= \lh -> return $
         setFormatter lh (tfLogFormatter "%Y-%m-%d %H:%M:%S%Q" "$time $msg")    
    updateGlobalLogger rootLoggerName (setHandlers [h] . setLevel INFO)
    Env{options = _opt, parties=_parties} <- if null (parsParties opt) 
        then do
            let _m = if (m opt < 0) then 1 else (m opt)
            when (_m > 1 && (myPid opt) == -1) $ do
                exPath <- getExecutablePath
                args <- getArgs
                forM_ [1..(_m-1)] $ \i -> do
                    let cmdLine = printf "%s %s -I %d" exPath (unwords args) i
                    createProcess (shell $
                        if os == "mingw32" then ("start " ++ cmdLine)
                        else if os == "darwin" then printf "osascript -e 'tell application \"Terminal\" to do script \"%s\"'" cmdLine
                        else "")
            
            let _mypid = if (myPid opt) >= 0 then (myPid opt) else 0
                _parties = map (\i -> mkParty i "127.0.0.1" ((basePort opt) + i)) [0.._m-1]
                _noASync = _m == 1 && ((noAsync opt) || not ((m opt) < 0))
            return Env{options = opt{myPid = _mypid, m = _m, noAsync = _noASync}, parties = _parties}
        else do
            let addresses = map (\addr -> splitOn ":" addr) (parsParties opt)
                _mypid = maybe (myPid opt) fromIntegral (elemIndex "" (map head addresses))
                _parties = map (\(i, [host, port]) -> mkParty i (if null host then "127.0.0.1" else host) (read port)) (zip [0..] addresses)
                _m =  (fromIntegral . length) _parties
            return Env{options = opt{myPid = _mypid, m = _m}, parties = _parties}

    let threshold = ((m _opt)-1) `div` 2
    return Env{options = _opt{threshold = threshold}, parties = _parties}
    where
        mkParty pid host port = Party{pid = pid, host = host, port = port}


-- secure types operator overloading
instance Num (SIO SecureTypes) where
    (*) = (.*)
    (+) = (.+)
    (-) = (.-)
    signum = ssignum False False
    fromInteger i =
         SecTypes.Literal <$> (liftIO $ newMVar $ FinFields.Literal i)

instance Fractional (SIO SecureTypes) where
    (/) = (./)
    recip = srecip

_coerce :: SecureTypes -> SecureTypes -> SecureTypes
_coerce (SecTypes.Literal _) f2 = f2
_coerce f1 _ = f1
    