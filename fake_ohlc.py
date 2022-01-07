def fake_ohlc(N=1000, start="2000/01/01", freq="D"):
    """
        Meh, need to make this better behaved
    """
    ind = pd.date_range(start, freq=freq, periods=N)
    returns = (np.random.random(N) - .5) * .05
    geom = (1+returns).cumprod()

    open = 100 * geom
    close = open + (open * (np.random.random(N) - .5)) * .1
    high = np.maximum(open, close) + .01
    low = np.minimum(open, close) - .01
    vol = 10000 + np.random.random(N) * 10000

    df = pd.DataFrame(index=ind)
    df['open'] = open
    df['high'] = high
    df['low'] = low
    df['close'] = close
    df['vol'] = vol.astype(int)
    return df
