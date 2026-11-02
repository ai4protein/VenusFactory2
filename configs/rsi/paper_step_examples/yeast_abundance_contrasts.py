"""Controller-authored numerical example; not an autonomous Scientist result."""
import math


def beta_fraction(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 501):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            return h
    raise ValueError('Incomplete beta did not converge')


def regularized_beta(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    weight = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                      + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        value = weight * beta_fraction(a, b, x) / a
    else:
        value = 1.0 - weight * beta_fraction(b, a, 1.0 - x) / b
    return max(0.0, min(1.0, value))


def reproduce(inputs):
    output = []
    for row in inputs['rows']:
        a = [float(v) for v in row['control'] if v is not None]
        b = [float(v) for v in row['treatment'] if v is not None]
        ma = sum(a) / len(a) if a else None
        mb = sum(b) / len(b) if b else None
        effect = mb - ma if a and b else None
        status, p = 'insufficient_replicates', None
        if len(a) == 3 and len(b) == 3:
            va = sum((v - ma) ** 2 for v in a) / 2
            vb = sum((v - mb) ** 2 for v in b) / 2
            sa, sb = va / 3, vb / 3
            if sa + sb > 0:
                df = (sa + sb) ** 2 / (sa * sa / 2 + sb * sb / 2)
                t2 = effect * effect / (sa + sb)
                p = regularized_beta(df / 2, 0.5, df / (df + t2))
                status = 'tested'
            else:
                status = 'zero_variance'
        output.append({'id': row['id'], 'n_control': len(a), 'n_treatment': len(b),
                       'mean_control': ma, 'mean_treatment': mb, 'log2_fold_change': effect,
                       'status': status, 'p_value': p, 'q_value': None,
                       'fourfold_p005': False, 'bh_fdr005': False})
    order = sorted([i for i, row in enumerate(output) if row['p_value'] is not None],
                   key=lambda i: output[i]['p_value'])
    running = 1.0
    for j in range(len(order) - 1, -1, -1):
        i = order[j]
        running = min(running, output[i]['p_value'] * len(order) / (j + 1))
        output[i]['q_value'] = running
        output[i]['fourfold_p005'] = output[i]['p_value'] < 0.05 and abs(output[i]['log2_fold_change']) >= 2
        output[i]['bh_fdr005'] = running < 0.05
    return {'contrast': inputs['contrast'], 'rows': output, 'tested': len(order)}
