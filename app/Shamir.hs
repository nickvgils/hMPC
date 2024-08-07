
-- TODO: Pseudorandom secret sharing (PRSS) allows one to share pseudorandom
-- secrets without any communication, as long as the parties
-- agree on a (unique) common public input for each secret.
-- PRSS relies on parties having agreed upon the keys for a pseudorandom
-- function (PRF).
{- |
Module for information-theoretic threshold secret sharing.

Threshold secret sharing assumes secure channels for communication.
-}
module Shamir (randomSplit, _recombinationVector, recombine, IdSharesPair) where

import Control.Monad
import FinFields
import System.Random
import Data.List

-- | Couples a ID pi to the share list si.
type IdSharesPair = (Integer, [Integer]) 



-- | Split each secret given in s into m random Shamir shares.
--
-- The (maximum) degree for the Shamir polynomials is t, @0 <= t < m@.
-- Return matrix of shares, one row per party.
randomSplit :: (RandomGen g) => FiniteField -> [FiniteField] -> Integer -> Integer -> g -> ([[Integer]], g)
randomSplit field s t m g = let (g', shares) = mapAccumR h g (map value s)   
                            in (transpose shares, g')
    where
        randoms gen len bound = let (gen1, gen2) = split gen
                                    _rands = take len $ randomRs (0, bound) gen1
                                in (_rands, gen2)
        h g' s_h = let (coefs, g'') = randoms g' (fromIntegral t) (modulus $ meta field)
                       shares = map (\i1 -> ((s_h + (poly coefs i1)) `mod` (modulus $ meta field))) [1..m]
                   in (g'', shares)           
        -- polynomial f(X) = s[h] + c[t-1] X + c[t-2] X^2 + ... + c[0] X^t
        poly c i1 = foldr (\c_j y -> (y + c_j) * i1) 0 c


-- | Compute and store a recombination vector.
--
-- A recombination vector depends on the field, the x-coordinates xs
-- of the shares and the x-coordinate x_r of the recombination point.
_recombinationVector :: FiniteField -> [Integer] -> Integer -> [Integer]
_recombinationVector field xs x_r = map coefs_div (zip [0..] xs)
    where
        coefs_div (i, x_i) = let (cf_n, cf_d) = coefs i x_i
                             in cf_n `div` cf_d
        coefs i x_i = foldr (\(j, x_j) (coef_n, coef_d) ->
            if j /= i
                then (coef_n * (x_r-x_j), coef_d * (x_i-x_j))
                else (coef_n, coef_d)) (1,1) (zip [0..] xs)
            
-- | Recombine shares given by points into secrets.
--
-- Recombination is done for x-coordinates x_rs.
recombine :: FiniteField -> [IdSharesPair] -> [FiniteField]
recombine field points =
    let (xs, shares) = unzip points
        vector = _recombinationVector field xs 0
        p = modulus $ meta field
    in map (\share_i -> field{value = (sum share_i vector) `mod` p}) (transpose shares)
        where
            sum share_i vector = foldr (\(i, s) -> (+) (s * (vector !! i) )) 0 (zip [0..] share_i)