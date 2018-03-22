data Gender = Male | Female deriving Eq

class Species s where
    gender :: s -> Gender

-- Returns true if s1 and s2 can conceive offspring
matable :: Species a => a -> a -> Bool
matable s1 s2 = gender s1 /= gender s2

data Human = Man | Woman
data Canine = Dog | Bitch

instance Species Human where
    gender Man = Male
    gender Woman = Female

instance Species Canine where
    gender Dog = Male
    gender Bitch = Female

bark Dog = "woof"
bark Bitch = "wow"

speak Man s = "The man says " ++ s
speak Woman s = "The woman says " ++ s
