module Parser (getArgParser, getArgParserExtra, Options(..)) where

import Options.Applicative

-- | Return parser results for command line arguments passed to the hMPC runtime.
getArgParser :: IO Options
getArgParser = execParser $ info (sample <**> helper) fullDesc

-- | Return parser for command line arguments passed to the hMPC runtime.
getArgParserExtra :: Parser a -- | include a user specified parser
    -> IO (Options, a)
getArgParserExtra pars = execParser $ info (((,) <$> sample <*> pars) <**> helper) fullDesc

data Options = Options
  { parsParties :: [String], m :: Integer, myPid :: Integer, 
    threshold :: Integer, basePort :: Integer, secParam :: Int, noAsync :: Bool, nrThreads :: Maybe Int,
    kernel :: [String], ghclib :: Maybe String, stack :: Bool} deriving (Show)

sample :: Parser Options
sample = Options
      <$> many (strOption
          ( short 'P' <> metavar "addr"
         <> help "use addr=host:port per party (repeat m times)" ))
      <*> option auto
          ( short 'M' <> showDefault <> value (-1)  <> metavar "INT"
         <> help "use m local parties (and run all m, if i is not set)" )
      <*> option auto
          ( short 'I' <> long "index" <> value (-1) <> metavar "INT"
         <> help "set index of this local party to i, 0<=i<m" )
      <*> option auto
          ( short 'T' <> long "threshold" <> showDefault <> value (-1) <> metavar "INT"
         <> help "threshold t, 0<=t<m/2" )
      <*> option auto
          ( short 'B' <> long "base-port" <> showDefault <> value 4242 <> metavar "b"
         <> help "use port number b+i for party i" )
      <*> option auto
          ( short 'K' <> long "sec-param" <> showDefault <> value 30 <> metavar "INT"
         <> help "security parameter k, leakage probability 2**-k" )
      <*> switch
          ( long "no-async" 
            <> help "disable asynchronous evaluation" )
      <*> optional ( option auto
          ( long "threads" <> metavar "L"
            <> help "Set the number of Haskell threads that can run truly simultaneously."))
      <*> many (argument str
          ( metavar "kernelkey" ))
      <*> optional (strOption
          (long "ghclib" <> metavar "INT"
         <> help "ghclib" ))
      <*> switch
          ( long "stack"
         <> help "Flag to indicate stack usage" )