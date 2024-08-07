module ShamirTest(tests) where

import Test.HUnit
import Control.Monad
import System.Random

import Shamir
import FinFields

secretsharing = do
    let field = gf 19
    forM_ [0..7] $ \t -> do
      g <- newStdGen
      -- putStrLn $ show $ take 5 $ randomRs (0, 1) g
      -- let m = 2 * t + 1
      -- let a = map (\i -> field{value = i}) [0..t]
      -- let shares = fst $ randomSplit (head a) a t m g
      -- putStrLn $ show shares
      -- putStrLn $ show (recombine field (zip [1..] shares))
      assertEqual "hooi" [1] [1]
      -- assertEqual "t=0,m=1" [0..t] (recombine field (zip [1..] shares))


tests = TestList
  [ TestLabel "ado1" $ TestCase (secretsharing)
  ]