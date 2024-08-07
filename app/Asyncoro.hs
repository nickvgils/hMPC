{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeApplications #-}

-- | This module provides basic support for asynchronous communication
-- and computation of secret-shared values.
module Asyncoro (createConnections, send, receive, Gather(..), async, asyncList, asyncListList, await, incPC, decreaseBarrier) where
import Network.Socket
import Network.Socket.ByteString (recv, sendAll)
import Control.Exception
import Control.Concurrent
import System.IO.Error
import Control.Monad
import Types
import Data.Function
import Data.List
import qualified Data.Map.Strict as Map
import qualified Data.Serialize as Enc
import qualified Data.ByteString as BS
import Data.Hashable
import Control.Monad.State
import SecTypes
import FinFields
import System.Log.Logger
import Text.Printf
import Parser


-- | Open connections with other parties, if any.
createConnections :: Int -> [Party] -> IO [Party]
createConnections myPid parties = do
    let m = length parties
    let listenPort = port $ parties !! myPid
    sock <- socket AF_INET Stream 0    -- create socket
    setSocketOption sock ReuseAddr 1   -- make socket immediately reusable - eases debugging.
    bind sock (SockAddrInet (fromIntegral listenPort) 0)   -- listen on TCP port 4242 + pid.
    listen sock 1                             -- set a max of 2 queued connections 
    serverParties <- replicateM myPid $ do
        mvar <- newEmptyMVar
        forkIO $ connectServer sock parties mvar
        return mvar
    
    clientParties <- forM (drop (myPid+1) parties) $ \party -> do
      mvar <- newEmptyMVar
      forkIO $ connectClient myPid party mvar
      return mvar

    channels <- mapM takeMVar (serverParties ++ clientParties)
    logging INFO $ printf "All %d parties connected." m

    close sock

    -- necessary if --single-threaded bug
    cap <- getNumCapabilities
    logging INFO $ printf "All threads run on %d cores." cap
    when (cap <= 1) (threadDelay 5000000)
    
    newDict <- newMVar Map.empty
    newMVar <- newMVar 0
    emptyChan <- newChan
    let selfParty = (parties !! myPid){outChan = emptyChan, sock = Nothing, dict = newDict, nbytesSent=newMVar}
    return $ sortBy (compare `on` pid) (selfParty: channels)

  where 
    connectServer sock parties mvar = do
      (conn, _) <- accept sock     -- accept a connection and handle it
      msg <- recv conn 1024 -- receive pid
      case Enc.decode msg of
        Right peer_pid -> initConnection conn (parties !! peer_pid) mvar

    connectClient pid peer mvar = do
      addr <- head <$> getAddrInfo Nothing (Just (host peer)) (Just $ show (port peer))
      sock <- openSocket addr
      res <- try @IOException $ connect sock (addrAddress addr)
      case res of
        Right _ -> do  -- connection successful
          sendAll sock (Enc.encode pid) --send pid
          initConnection sock peer mvar
        Left _ -> do  -- exception
          threadDelay 100000
          connectClient pid peer mvar

    initConnection sock peer mvar = do
      newDict <- newMVar Map.empty
      bytesMvar <- newMVar 1 -- one for the runSession to complete
      outChan <- newChan
      putMVar mvar peer{outChan = outChan, sock = Just sock, dict = newDict, nbytesSent=bytesMvar}
      runConnection outChan sock newDict bytesMvar


-- read lines from the socket and insert into dictionary
runConnection :: Chan BS.ByteString -> Socket -> MVar Dict -> MVar Int -> IO ()
runConnection chan sock dictMvar nbytesSent = do
    reader <- forkIO $ forever $ do
        dataToSend <- readChan chan
        sendAll sock dataToSend

    handle (\(SomeException _) -> return ()) $ flip fix BS.empty $ \loop buffer_old ->
      BS.append buffer_old <$> recv sock 1024 
        >>= decodeMessageChecks dictMvar nbytesSent 
        >>= loop
        >> return ()
    
    killThread reader

    where 
      decodeMessageChecks dictMvar nbytesSent buffer = do
        let bufferLength = BS.length buffer
        if bufferLength < 4
          then return buffer
          else do
            let payload_length = _decode Enc.getInt32le (BS.take 4 buffer)
                len_packet = payload_length + 12
            if bufferLength < len_packet
              then return buffer
              else decodeMessage buffer len_packet

      decodeMessage buffer len_packet = do
        let (msg, leftover) = BS.splitAt len_packet buffer
            pc = _decode Enc.getInt64le (BS.drop 4 buffer)
        modifyMVar_ nbytesSent (return . (+ len_packet))
        modifyMVar_ dictMvar $ \dict ->
          case Map.lookup pc dict of
            Just mvar -> do
              putMVar mvar (BS.drop 12 msg)
              return $ Map.delete pc dict
            Nothing -> do
              mvar <- newMVar (BS.drop 12 msg)
              return $ Map.insert pc mvar dict                                                    
        decodeMessageChecks dictMvar nbytesSent leftover

      _decode decoder buffer = 
        case Enc.runGet (liftM fromIntegral decoder) buffer of
          Right t -> t

-- | Receive payload labeled with given pc from the peer.
receive :: Int -> Party -> SIO (MVar BS.ByteString)
receive pc party = liftIO $ modifyMVar (dict party) $ \dict -> do
    case Map.lookup pc dict of
        Just value -> return (Map.delete pc dict, value)
        Nothing -> do
            mvar <- newEmptyMVar
            return ((Map.insert pc mvar dict), mvar)

-- | Transform 'SecureTypes' into 'FiniteField' by reading the future 'MVar' share that contains a 'FiniteField' (blocking).
class Gather a where
  type Result a :: *
  gather :: a -> SIO (Result a)

instance Gather SecureTypes where
  type Result SecureTypes = FiniteField
  gather = await . share

instance Gather a => Gather [a] where
  type Result [a] = [Result a]
  gather = mapM gather

instance (Gather a, Gather b) => Gather (a, b) where
  type Result (a, b) = (Result a, Result b)
  gather (x, y) = do
    resultX <- gather x
    resultY <- gather y
    return (resultX, resultY)

instance (Gather a, Gather b, Gather c) => Gather (a, b, c) where
  type Result (a, b, c) = (Result a, Result b, Result c)
  gather (x, y, z) = do
    resultX <- gather x
    resultY <- gather y
    resultZ <- gather z
    return (resultX, resultY, resultZ)

-- | Read the value from the future MVar (blocking).
await :: MVar a -> SIO a
await = liftIO . readMVar


-- | Send payload labeled with pc to the peer.
--
-- Message format consists of three parts:
--
-- 1. pc (8 bytes signed int)
--
-- 2. payload_size (4 bytes unsigned int)
-- 
-- 3. payload (byte string of length payload_size).
send :: Int -> BS.ByteString -> Party -> SIO ()
send pc payload party = do
  let payload_size = (BS.length payload)
  let bytes = ((Enc.runPut ( 
        (Enc.putInt32le . fromIntegral) payload_size
        >> (Enc.putInt64le . fromIntegral) pc)) <> payload)
  liftIO $ writeChan (outChan party) bytes

-- | increment program counter in state.
incPC :: SIO Int
incPC = do
    pcOld <- gets pc
    modify (\env -> env{pc = (+1) pcOld})
    gets pc

-- | 'forkIO' the action monad asynchronously and return future 'MVar'.
-- Provide the given state monad with its own program counter space.
async :: SIO a -> SIO (MVar a)
async = \action -> head <$> asyncList 1 ((:[]) <$> action)

asyncList :: Int -> SIO [a] -> SIO [MVar a]
asyncList l = \action -> head <$> asyncListList 1 l ((:[]) <$> action)

asyncListList :: Int -> Int -> SIO [[a]] -> SIO [[MVar a]]
asyncListList l1 l2 = \action -> do
    pcOld <- incPC
    state <- get
    outslist <- replicateM l1 $ replicateM l2 (liftIO $ newEmptyMVar)

    let newState = state{pc = hash $ show pcOld}
        barrier = forkIOBarrier state
        action2 = do
          fieldslist <- runSIO action newState
          zipWithM_ (zipWithM_ putMVar) outslist fieldslist
          decreaseBarrier barrier    

    liftIO $ modifyMVar_ (count barrier) (return . (+1))
    if (noAsync . options) state
      then liftIO $ action2
      else void $ liftIO $ forkIO $ action2

    return outslist 

decreaseBarrier :: Barrier -> IO ()
decreaseBarrier (Barrier countVar signalVar) =
          modifyMVar_ countVar $ \n -> do
            let n' = n - 1
            if n' == 0
              then do
                putMVar signalVar ()
                return n'
              else return n'