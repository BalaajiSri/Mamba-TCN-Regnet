from utils.actions import BUY, HOLD, SELL


def buy_sell_smart(today, pred, balance, shares, risk=5):
    diff = pred * risk / 100
    if today > pred + diff:
        balance += shares * today
        shares = 0
    elif today > pred:
        factor = (today - pred) / diff
        balance += shares * factor * today
        shares *= (1 - factor)
    elif today > pred - diff:
        factor = (pred - today) / diff
        shares += balance * factor / today
        balance *= (1 - factor)
    else:
        shares += balance / today
        balance = 0
    return balance, shares

def buy_sell_smart_w_short(today, pred, balance, shares, risk=5, max_n_btc=0.002):
    diff = pred * risk / 100
    if today < pred - diff:
        shares += balance / today
        balance = 0
    elif today < pred:
        factor = (pred - today) / diff
        shares += balance * factor / today
        balance *= (1 - factor)
    elif today < pred + diff:
        if shares > 0:
            factor = (today - pred) / diff
            balance += shares * factor * today
            shares *= (1 - factor)
    else:
        balance += (shares + max_n_btc) * today
        shares = -max_n_btc
    return balance, shares

def buy_sell_vanilla(today, pred, balance, shares, tr=0.01):
    tmp = abs((pred - today) / today)
    if tmp < tr:
        return balance, shares
    if pred > today:
        shares += balance / today
        balance = 0
    else:
        balance += shares * today
        shares = 0
    return balance, shares


def apply_action(
    action,
    today,
    balance,
    shares,
    allow_short=False,
    max_short=0.002,
):
    if action == BUY:
        if shares < 0:
            balance += shares * today
            shares = 0
        shares += balance / today
        balance = 0
        return balance, shares

    if action == SELL:
        balance += shares * today
        shares = 0
        if allow_short:
            balance += max_short * today
            shares = -max_short
        return balance, shares

    return balance, shares


def trade(
    data,
    time_key,
    timstamps,
    targets,
    preds,
    balance=100,
    mode='smart_v2',
    risk=5,
    y_key='Close',
    step_seconds=86400,
    actions=None,
    allow_short=False,
    current_prices=None,
):
    balance_in_time = [balance]
    shares = 0

    action_iterable = actions if actions is not None else [None] * len(timstamps)
    current_price_iterable = current_prices if current_prices is not None else [None] * len(timstamps)
    for ts, target, pred, action, current_price in zip(
        timstamps,
        targets,
        preds,
        action_iterable,
        current_price_iterable,
    ):
        target_row = data[data[time_key] == int(ts)]
        if target_row.empty:
            raise ValueError(f"Missing target timestamp {int(ts)} in trade data.")
        assert round(target, 2) == round(target_row.iloc[0][y_key], 2)

        if current_price is None:
            current_row = data[data[time_key] == int(ts - step_seconds)]
            if current_row.empty:
                raise ValueError(
                    f"Missing previous timestamp {int(ts - step_seconds)} for target timestamp {int(ts)}. "
                    "Pass current_prices from the model batch or provide data with the previous row included."
                )
            today = current_row.iloc[0][y_key]
        else:
            today = current_price

        if action is not None:
            balance, shares = apply_action(
                action=action,
                today=today,
                balance=balance,
                shares=shares,
                allow_short=allow_short,
            )
        elif mode == 'smart':
            balance, shares = buy_sell_smart(today, pred, balance, shares, risk=risk)
        elif mode == 'smart_w_short':
            balance, shares = buy_sell_smart_w_short(today, pred, balance, shares, risk=risk, max_n_btc=0.002)
        elif mode == 'vanilla':
            balance, shares = buy_sell_vanilla(today, pred, balance, shares)
        elif mode == 'no_strategy':
            shares += balance / today
            balance = 0
        balance_in_time.append(shares * today + balance)

    balance += shares * targets[-1]
    return balance, balance_in_time
