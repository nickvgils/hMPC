hMPC basics
===========

The goal of hMPC is that you can render nice and useful programs
with relative ease. The complexities of the secure multiparty computation
protocols running "under the hood" are hidden as much possible.

In this brief tutorial we show how to use hMPC in a Haskell program, focusing
on the specific steps you need to do (extra) to implement a multiparty computation.

Loading hMPC
------------

To load hMPC we use the following ``import`` statement:

.. code-block:: haskell

    import Runtime as Mpc

To set up the hMPC runtime:

.. code-block:: haskell

    Mpc.runMpc $ do

which means that the identities and addresses of
the MPC parties have been loaded. The details of these parties can be inspected via
``gets parties``.

Secure types
------------

To perform any meaningful multiparty computation in hMPC we'll first need to create
appropriate secure types. For example, to work with secure integers we may create a secure
type ``SecInt`` as follows:

.. code-block:: haskell

    secInt <- Mpc.secIntGen (16)

Unlike Haskell integers (type ``Integer``), secure integers always have a maximum bit length.
Here, we use 16-bit (signed) integers. The limited range of values ensures that "under the hood"
a secure integer can be represented by an element of a finite field.
More precisely, a secure integer is actually stored in secret-shared form, where each party
will hold exactly one share represented as an element of a field of prime order.

.. You can create as many secure types as you like, mixing secure integers, secure fixed-point numbers,
.. secure floating-point numbers, and even secure types for (elements of) finite fields and some classes
.. of finite groups. If you're curious, run ``python -m mpyc`` from the command line to get a list
.. of secure types to play with.

Secure input
------------

To let all parties provide their age as a private input, use:

.. code-block:: haskell

    myAge <- putStrLn "Enter your age: " >> readLn  # each party enters a number
    ourAges <- Mpc.input (secInt myAge)  # list with one secint per party

Each party runs its own copy of this code, so ``myAge`` will only be known to the party entering the number.
The value for ``myAge`` is then converted to a secure integer to tell ``Mpc.input`` the type for secret-sharing
the value of ``myAge``.

The list ``ourAges`` will contain the secret-shares of the ages entered by all parties,
represented by one secure integer of type ``SecInt`` per party.

Secure computation
------------------

We perform some computations to determine the total age, the maximum age, and the number of ages above average:

.. code-block:: haskell

    totalAge <- return <$> sum ourAges    
    let maxAge = Mpc.smaximum ourAges
    m <- length <$> (gets parties)
    let aboveAvg = sum $ map (\age -> Mpc.ifElse ((age * fromIntegral m) .> totalAge) 1 0) ourAges

For the total age we can use the Haskell function ``sum``, although ``Mpc.sum`` would be slightly faster.
For the maximum age we cannot use the Haskell function ``maximum``, so we use ``Mpc.smaximum``.
To compute the number of ages above average, we compare each ``age`` with the average age ``totalAge / m``,
however, avoiding the use of a division.

As the result of a comparison with secure integers is a secret-shared bit, we can get the result more
directly:

.. code-block:: haskell

    let aboveAvg = Mpc.ssum $ map (\age -> (age * fromIntegral m) .> totalAge) ourAges

This time also using ``Mpc.ssum`` instead of ``sum`` for slightly better performance.

Secure output
-------------

Finally, we reveal the results:

.. code-block:: haskell

    liftIO . putStrLn . ("Average age: " ++) . show 
        . (\x -> (fromIntegral x) / (fromIntegral m))
            =<< await =<< Mpc.output totalAge
    liftIO . putStrLn . ("Maximum age: " ++) . show
        =<< await =<< Mpc.output maxAge
    liftIO . putStrLn . ("Number of \"elderly\": " ++) . show
        =<< await =<< Mpc.output aboveAvg

Note that we need to ``await`` the results of the calls to ``Mpc.output``.

Running hMPC
------------

To run the above code with multiple parties, we put everything together,
inserting call ``Mpc.runSession $ do`` to let the
parties actually connect and disconnect:

.. code-block:: haskell
    :caption: Elderly.hs

    import Runtime as Mpc

    main :: IO ()
    main = Mpc.runMpc $ do
        secInt <- Mpc.secIntGen 16

        Mpc.runSession $ do
            myAge <- liftIO $ putStrLn "Enter your age: " >> readLn
            ourAges <- Mpc.input (secInt myAge)

            totalAge <- return <$> Mpc.ssum ourAges
            let maxAge = Mpc.smaximum ourAges
            m <- length <$> (gets parties)
            let aboveAvg = Mpc.ssum $ map (\age -> (age * fromIntegral m) .> totalAge) ourAges

            liftIO . putStrLn . ("Average age: " ++) . show 
                . (\x -> (fromIntegral x) / (fromIntegral m))
                    =<< await =<< Mpc.output totalAge
            liftIO . putStrLn . ("Maximum age: " ++) . show
                =<< await =<< Mpc.output maxAge
            liftIO . putStrLn . ("Number of \"elderly\": " ++) . show
                =<< await =<< Mpc.output aboveAvg

An example run between three parties on `localhost` looks as follows:

.. code-block::

    $ cabal run your-executable-name -- -M3 -I0 --no-log
    Enter your age: 21
    Average age: 29.0
    Maximum age: 47
    Number of "elderly": 1

.. code-block::

    $ cabal run your-executable-name -- -M3 -I1 --no-log
    Enter your age: 19
    Average age: 29.0
    Maximum age: 47
    Number of "elderly": 1

.. code-block::

    $ cabal run your-executable-name -- -M3 -I2 --no-log
    Enter your age: 47
    Average age: 29.0
    Maximum age: 47
    Number of "elderly": 1

.. See :ref:`MPyC demos <mpyc demos>` for lots of other examples, including
.. some more elaborate explanations in Jupyter notebooks.
