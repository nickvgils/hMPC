-- | This module collects all hGMP functions used by hMPC.
module Hgmp (isPrime, prevPrime, invert) where

import Numeric.GMP.Utils (withInInteger, withOutInteger_)
import Numeric.GMP.Raw.Safe
import System.IO.Unsafe (unsafePerformIO)


-- | Return True if x is probably prime, else False if x is
-- definitely composite
isPrime :: Integer -> Bool
isPrime x = unsafePerformIO (withInInteger x (flip mpz_probab_prime_p 25)) > 0


-- | Return the greatest probable prime number < x, if any.
prevPrime :: Integer -> Integer
prevPrime x
    | x <= 2 = 0
    | x == 3 = 2
    | otherwise = _prevPrime (x - (1 + (x `mod` 2)))
    where
        _prevPrime x
            | isPrime x = x
            | otherwise = _prevPrime (x-2)

-- | Return y such that @x*y == 1 modulo m@.
invert :: Integer -> Integer -> Integer
invert x m = unsafePerformIO $
                withOutInteger_ $ \rop ->
                    withInInteger x $ \op1 ->
                        withInInteger m $ \op2 ->
                            mpz_invert rop op1 op2