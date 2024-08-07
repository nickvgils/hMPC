module Types where

import Control.Concurrent
import Network.Socket
import qualified Data.Map.Strict as Map
import Control.Monad.State
import Parser
import qualified Data.ByteString as BS
import System.Log.Logger
import System.Random
import Data.Time.Clock

-- read dictionary, key = pc, value = mvar bytestring
type Dict = Map.Map Int (MVar BS.ByteString)
-- pid, (socket send to peer (nothing if self), read value from mvar in dict)
data Party = Party {pid :: Integer, host :: String, port :: Integer, outChan :: Chan BS.ByteString, sock :: Maybe Socket, dict :: MVar Dict, nbytesSent :: MVar Int}

data Barrier = Barrier { count :: MVar Int, signal :: MVar () }

-- state transformer environment
data Env = Env {
    parties :: [Party],
    pc :: Int,
    options :: Options,
    forkIOBarrier :: Barrier,
    gen :: StdGen,
    startTime :: UTCTime
}

type SIO a = StateT Env IO a

runSIO :: SIO a -> Env -> IO a
runSIO = evalStateT

logging :: Priority -> String -> IO ()
logging prio str = logM rootLoggerName prio str