{- 
Demo decision tree learning using ID3.

This demo implements Protocol 4.1 from the paper 'Practical Secure Decision
Tree Learning in a Teletreatment Application' by Sebastiaan de Hoogh, Berry
Schoenmakers, Ping Chen, and Harm op den Akker, which appeared at the 18th
International Conference on Financial Cryptography and Data Security (FC 2014),
LNCS 8437, pp. 179-194, Springer.
See https://doi.org/10.1007/978-3-662-45472-5_12 (or,
see https://fc14.ifca.ai/papers/fc14_submission_103.pdf or,
see https://www.researchgate.net/publication/295148009).

ID3 is a recursive algorithm for generating decision trees from a database
of samples (or, transactions). The first or last attribute of each sample
is assumed to be the class attribute, according to which the samples are
classified. Gini impurity is used to determine the best split. See also
https://en.wikipedia.org/wiki/ID3_algorithm.

The samples are assumed to be secret, while the resulting decision tree is
public. All calculations are performed using secure integer arithmetic only.

The demo includes the following datasets (datasets 1-4 are used in the paper):

  0=tennis: classic textbook example
  1=balance-scale: balance scale weight & distance database
  2=car: car evaluation database
  3=SPECT: Single Proton Emission Computed Tomography train+test heart images
  4=KRKPA7: King+Rook versus King+Pawn-on-A7 chess endgame
  5=tic-tac-toe: Tic-Tac-Toe endgame
  6=house-votes-84: 1984 US congressional voting records

The numbers 0-6 can be used with the command line option -i of the demo.

Three command line options are provided for controlling accuracy:

  -l L, --bit-length: overrides the preset bit length for a dataset
  -e E, --epsilon: minimum number of samples E required for attribute nodes,
                   represented as a fraction of all samples, 0.0<=E<=1.0
  -a A, --alpha: scale factor A to avoid division by zero when calculating
                 Gini impurities, basically, by adding 1/A to denominators, A>=1

Setting E=1.0 yields a trivial tree consisting of a single leaf node, while
setting E=0.0 yields a large (overfitted) tree. Default value E=0.05 yields
trees of reasonable complexity. Default value A=8 is sufficiently large for
most datasets. Note that if A is increased, L should be increased accordingly.

Finally, the command line option --parallel-subtrees can be used to let the
computations of the subtrees of an attribute node be done in parallel. The
default setting is to compute the subtrees one after another (in series).
The increase in the level of parallelism, however, comes at the cost of
a larger memory footprint.

Interestingly, one will see that the order in which the nodes are added to
the tree will change if the --parallel-subtrees option is used. The resulting
tree is still the same, however. Of course, this only happens if the hMPC
program is actually run with multiple parties (or, if the -M1 option is used).
-}
module Id3gini (main) where

import Paths_hMPC (getDataFileName)
import Control.Monad.State (liftIO)
import Control.Monad (forM)
import Data.List (transpose, isPrefixOf)
import Data.List.Split (splitOn)
import Data.Char (toLower)
import SecTypes (secIntGen, SecureTypes)
import System.Log.Logger (infoM, rootLoggerName)
import Text.Printf (printf)
import Options.Applicative
import Types
import Runtime as Mpc
import qualified Data.Set as Set


data Tree = Leaf Int | Node Int [Tree] deriving (Show)

data Env = Env {
  ms :: [[[SIO SecureTypes]]],
  c :: Int,
  opts :: Options
}

data Options = Options
  { dataset :: Int, bitLength :: Maybe Int, epsilon :: Float, 
    alpha :: Int, parallelSubtrees :: Bool, noPrettyTree :: Bool }


id3 :: Id3gini.Env -> Set.Set Int -> [SIO SecureTypes] -> SIO Tree
id3 env r vt = do
  sizes <- mapM (return <$> (Mpc.inProd vt)) ((ms env) !! (c env))
  (i , mx) <- Mpc.argmax sizes
  sizeT <- return <$> Mpc.ssum sizes
  let stop = (sizeT .<= (fromIntegral . floor) ((epsilon $ opts env) * (fromIntegral $ length vt))) + (mx .== sizeT) 
  isZero <- Mpc.await =<< Mpc.isZeroPublic stop
  if not $ (not . null) r && isZero
    then do
      i <- Mpc.await =<< Mpc.output i
      liftIO $ infoM rootLoggerName ("Leaf node label " ++ show i)
      return $ Leaf $ fromInteger i
    else do
      vt_sc <- mapM (Mpc.schurProd vt) ((ms env) !! (c env))
      gains <- forM (Set.toList r) $ \a -> 
        Mpc.matrixProd (ms env !! a) vt_sc True >>= gi (alpha $ opts env)
      k <- Mpc.await =<< Mpc.output =<< fst <$> (Mpc.argmaxfunc gains secureFraction)
      let a = (Set.toList r) !! fromInteger k
      vt_sa <- mapM (Mpc.schurProd vt) ((ms env) !! a)
      liftIO $ infoM rootLoggerName ("Attribute node " ++ show a)
      subtrees <- if parallelSubtrees $ opts env
          then mapM (async . id3 env (Set.difference r (Set.fromList [a]))) vt_sa 
                >>= mapM Mpc.await
          else mapM (id3 env (Set.difference r (Set.fromList [a]))) vt_sa
       
      return $ Node a subtrees

gi :: Int -> [[SIO SecureTypes]] -> SIO [SIO SecureTypes]
gi alpha x = do
  y <- forM x $ \a -> return <$> ((fromIntegral alpha) * (Mpc.ssum a) + 1)
  d <- return <$> Mpc.sproduct y
  let g = Mpc.inProd (zipWith Mpc.inProd x x) (map (1 /) y)
  return [d * g, d]

secureFraction :: [SIO SecureTypes] -> [SIO SecureTypes] -> SIO SecureTypes
secureFraction [n1, d1] [n2, d2] = Mpc.inProd [n1, negate d1] [d2, n2] .< 0


depth :: Tree -> Int
depth (Leaf _) = 0
depth (Node _ subtrees) = 1 + maximum (map depth subtrees)

size :: Tree -> Int
size (Leaf _) = 1
size (Node _ subtrees) = 1 + sum (map size subtrees)

pretty :: String -> Tree -> [String] -> [[String]] -> Int -> String
pretty prefix (Leaf a) names ranges c = (ranges !! c) !! a
pretty prefix (Node a subtrees) names ranges c =
  let subtreeStrings = map (\(s, t) -> 
        printf "\n%s%s == %s: %s" prefix (names !! a) s (pretty ("|   " ++ prefix) t names ranges c)) 
          (zip (ranges !! a) subtrees)
  in concat subtreeStrings

settings = [("tennis", 32), ("balance-scale", 77), ("car", 95),
                ("SPECT", 42), ("KRKPA7", 69), ("tic-tac-toe", 75), ("house-votes-84", 62)]

parser :: Parser Options
parser = Options
      <$> option auto
          ( short 'i' <> long "dataset" <> showDefault <> value 0  <> metavar "I"
            <> help ("dataset 0=tennis (default), 1=balance-scale, 2=car, \
                        \3=SPECT, 4=KRKPA7, 5=tic-tac-toe, 6=house-votes-84"))
      <*> optional ( option auto
          ( short 'l' <> long "bit-length"  <> metavar "L"
            <> help "override preset bit length for dataset"))
      <*> option auto
          ( short 'e' <> long "epsilon" <> showDefault <> value 0.05  <> metavar "E"
            <> help "minimum fraction E of samples for a split, 0.0<=E<=1.0")
      <*> option auto
          ( short 'a' <> long "alpha" <> showDefault <> value 8  <> metavar "A"
            <> help "scale factor A to prevent division by zero, A>=1")
      <*> switch
          ( long "parallel-subtrees" 
            <> help "process subtrees in parallel (rather than in series)" )
      <*> switch
          ( long "no-pretty-tree" 
            <> help "print raw flat tree instead of pretty tree" )

main :: IO ()
main = Mpc.runMpcWithArgs parser $ \opts -> do
  let (name, bitLengthOpts) = settings !! (dataset opts)
  secInt <- secIntGen (maybe bitLengthOpts id (bitLength opts))

  filePath <- liftIO $ getDataFileName $ "id3/" ++ name ++ ".csv"
  content <- liftIO $ readFile filePath
  let (attrNames, transactions) = splitAt 1 $ map (splitOn ",") $ lines content
      columns = transpose transactions
      attrRanges = map (Set.toList . Set.fromList) columns
      (n, d) = (length transactions, length $ head $ transactions)
      checkPrefix prefix len = if isPrefixOf "class" prefix then 0 else len - 1
      vars2 = Id3gini.Env {
        -- one-hot encoding of attributes:
        ms = (map (\(attrRange, col) -> 
              map (\attr -> 
              map (\elem -> (secInt . fromIntegral . fromEnum) (attr == elem)) 
                      col) attrRange) (zip attrRanges columns)),
        c = checkPrefix (map toLower (head $ head attrNames)) (length $ head attrNames),
        opts = opts }
      vt = replicate n (secInt 1)
      r = Set.difference (Set.fromList [0..(d - 1)]) (Set.fromList [c vars2])

  liftIO $ printf "dataset: %s with %d samples and %d attributes\n" name n (d-1)
  tree <- Mpc.runSession $ id3 vars2 r vt
  liftIO $ printf "Decision tree of depth %d and size %d: " (depth tree) (size tree)
  liftIO $ putStrLn $ if noPrettyTree opts 
    then show tree 
    else pretty "if " tree (head attrNames) attrRanges (c vars2)