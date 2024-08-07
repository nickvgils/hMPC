module Main (main) where

import ShamirTest
import RuntimeTest
import Test.HUnit

main :: IO ()
main = do
  runTestTT allTests
  return ()

allTests :: Test
allTests = TestList
  [ 
    TestLabel "ShamirThresha" ShamirTest.tests,
    TestLabel "Runtime" RuntimeTest.tests
  ]