module MpcTools where



-- reduce :: (SIO SecureTypes -> SIO SecureTypes -> SIO SecureTypes) -> [SIO SecureTypes] -> Maybe (SIO SecureTypes) -> SIO SecureTypes
-- reduce _ [] (Just iv) = iv
-- reduce _ [] Nothing = error "reduce: empty list with no initial value"
-- reduce _ [x] _ = x
-- reduce f xs iv =
--     let reduce' [] = []
--         reduce' [y] = [y]
--         reduce' (y1:y2:ys) = f y1 y2 : reduce' ys
--     in head (reduce' xs')
--     where xs' = maybe xs (:xs) iv