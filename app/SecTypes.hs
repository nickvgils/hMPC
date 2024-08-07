{- |
This module collects basic secure (secret-shared) types for hMPC.

Secure number types all use common base classes, which
ensures that operators such as +,* are defined by operator overloading.
-}
module SecTypes where

import Hgmp
import Data.Bits
import Types
import Control.Concurrent.MVar
import Control.Monad.State
import FinFields
import Parser

-- | A secret-shared object.
--
-- An MPC protocol operates on secret-shared objects of type SecureObject.
-- The basic Haskell operators are overloaded instances by SecureTypes classes.
-- An expression like a * b will create a new SecureObject, which will
-- eventually contain the product of a and b. The product is computed
-- asynchronously, using an instance of a specific cryptographic protocol.
data SecureTypes = 
    -- | Base class for secure (secret-shared) numbers.
    SecFld {field :: FiniteField, share :: MVar FiniteField, bitLength :: Int}
    -- | Base class for secure (secret-shared) finite field elements.
    | SecInt {field :: FiniteField, share :: MVar FiniteField, bitLength :: Int}
    | Literal {share :: MVar FiniteField}

-- | Secure l-bit integers ('SecInt').
secIntGen :: Int -> SIO (Integer -> SIO SecureTypes)
secIntGen l = do
    k <- secParam <$> gets options
    let field = FinFields.gf (Hgmp.prevPrime $ 1 `shiftL` (l + k + 2))
    return $ setShare $ SecInt {field = field, bitLength = l}


-- | Secure finite field ('SecFld') of order q = p
-- where p is a prime number
secFldGen :: Integer -> (Integer -> SIO SecureTypes)
secFldGen pnew = let field = FinFields.gf pnew
                     bitLength = (ceiling . logBase 2 . fromIntegral) pnew
                 in setShare $ SecFld {field = field, bitLength = bitLength}

setShare :: SecureTypes -> Integer -> SIO SecureTypes
setShare sectype val = do
    mvar <- liftIO $ newMVar (field sectype){value = val}
    return $ sectype {share = mvar}