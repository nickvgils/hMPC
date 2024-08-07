module RuntimeTest(tests) where

import Test.HUnit
-- import Control.Concurrent.MVar

-- import Shamir
import Runtime
-- import SecTypes
-- import FinFields
import Control.Monad.State
-- import Data.Hashable



runtimeFunctions = runMpc $ do
    secInt <- secIntGen 64
    let secFld = secFldGen 101
        (af, bf) = (secFld 6, secFld 6)
        (a, b) = (secInt (14), secInt 14)
        (v1, v2) = (map secInt [1,8,3,10,2,7], map secInt [4,5,6])
        (amat, bmat) = ([v1,v2], (map . map) secInt [[10,11], [20,21], [30,31]])

        n1 = 5
        n2 = 3
        k = 11
    
    runSession $ do
        -- out <- Runtime.sisZeroPublic a
        -- out <- await =<< Runtime.output (a .<= b)
        out <- Runtime.await =<< Runtime.output (fst =<< argmax v1)
        -- out <- Runtime.output $ smaximum v1
        -- out <- Runtime.output =<< ifElseList (a .< b) [b] [a]
        -- out <- Runtime.output $ Runtime.ssignum True False a
        -- out <- Runtime.output . concat =<< Runtime.matrixProd amat bmat False
        -- out <- Runtime.output $ Runtime.inProd v1 v2
        -- out <- Runtime.output =<< Runtime.schurProd v1 v2
        -- inp <- Runtime.input v1
        -- out <- Runtime.output (inp !! 0)
        -- out <- Runtime.output $ Runtime.sproduct v1
        -- out <- Runtime.output =<< Runtime.randomBits a 5 True
        -- out <- Runtime.output (af .== bf)
        -- out <- Runtime.output (a / b)
        -- out <- Runtime.output $ recip a
        -- out <- Runtime.output ((a *a) * (a * a))
        -- out <- Runtime.output (a .^ k)
        liftIO $ putStrLn $ "outcome var " ++ show k ++ " : " ++ show out
        return ()

tests = TestList
  [ 
    TestLabel "runtime12" $ TestCase (runtimeFunctions)
  ]