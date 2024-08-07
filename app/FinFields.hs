{-# LANGUAGE TemplateHaskell #-}

-- TODO
-- Taking square roots and quadratic residuosity tests supported as well.
-- Moreover, (multidimensional) arrays over finite fields are available
-- with operators +,-,*,/ for convenient and efficient NumPy-based
-- vectorized processing next to operator @ for matrix multiplication.
-- Much of the NumPy API can be used to manipulate these arrays as well.

{- |
This module supports finite (Galois) fields.

Function 'gf' creates types implementing finite fields.
-}
module FinFields (FiniteField(..), FiniteFieldMeta(..), gf, toBytes, fromBytes) where

import Hgmp as Hgmp
import Data.Bits
import qualified Data.ByteString as B
import Data.List.Split


-- | Instantiate an object from a field and subsequently apply overloaded
-- operators such as @('+')@, @('-')@, @('*')@, @('/')@ etc., to compute with field elements.
data FiniteField =  FiniteField {meta :: FiniteFieldMeta, value :: Integer} 
                    | Literal {value :: Integer} deriving Show
data FiniteFieldMeta = FiniteFieldMeta {modulus :: Integer, byteLength :: Int} deriving Show


-- TODO: irreducible polynomial, also creates corresponding array type.
-- | Create a finite (Galois) field for given modulus (prime number).
gf :: Integer -> FiniteField
gf modulus = let byteLength = ((ceiling . logBase 2 . fromIntegral) modulus + 7) `shiftR` 3
                in FiniteField{meta = FiniteFieldMeta {modulus = modulus, byteLength = byteLength}}

-- | Multiplicative inverse, Division.
instance Fractional FiniteField where
    recip a = a{value = Hgmp.invert (value a) (modulus $ meta a)}
    (/) a b = a * recip (case b of Literal val -> a{value = val}; _ -> b)

-- | Equality test.
instance Eq FiniteField where
    (==) f1 f2 = (value f1) == (value f2)

-- | Addition, Subtraction, Multiplication.
instance Num FiniteField where
    (+) = _applyOpFF (+)
    (-) = _applyOpFF (-)
    (*) = _applyOpFF (*)
    fromInteger a = Literal a

_applyOpFF :: (Integer -> Integer -> Integer) -> FiniteField -> FiniteField -> FiniteField
_applyOpFF f (FiniteField meta val1) (FiniteField _ val2) = _makeFF meta (f val1 val2)
_applyOpFF f (FiniteField meta val1) (Literal val2) = _makeFF meta (f val1 val2)
_applyOpFF f (Literal val1) (FiniteField meta val2) = _makeFF meta (f val1 val2)
_applyOpFF f (Literal val1) (Literal val2) = Literal (f val1 val2)

_makeFF :: FiniteFieldMeta -> Integer -> FiniteField
_makeFF meta val = FiniteField meta (val `mod` modulus meta)

-- | Return byte string representing the given list/ndarray of integers x.
toBytes :: Int -> [Integer] -> B.ByteString
toBytes byteLength x = B.pack $ concatMap (\a -> map (\i -> fromIntegral ((a `shiftR` (i * 8)) .&. 0xFF)) [0..byteLength-1]) x

-- | Return the list of integers represented by the given byte string.
fromBytes :: Int -> B.ByteString -> [Integer]
fromBytes n bytes = map (foldr unstep 0) (chunksOf n $ B.unpack bytes)
  where
    unstep b a = a `shiftL` 8 .|. fromIntegral b



