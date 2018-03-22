data Gender = Male | Female deriving Eq

class Species s where
    gender :: s -> Gender

-- Returns true if s1 and s2 can conceive offspring
matable :: Species a => a -> a -> Bool
matable s1 s2 = gender s1 /= gender s2

data Human = Man | Woman
data Canine = MaleDog | FemaleDog

instance Species Human where
    gender Man = Male
    gender Woman = Female

instance Species Canine where
    gender MaleDog = Male
    gender FemaleDog = Female

bark MaleDog = "woof"
bark FemaleDog = "wow"

speak Man s = "The man says " ++ s
speak Woman s = "The woman says " ++ s
