import numpy as np

def shuffle(*arrays, **kwargs): #arrays：(items,positems,negitems) ,这里真没看明白kwargs有什么用，感觉没用

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:#确保三个输入都是同维的
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)#给user重新洗牌

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)#拿shuffle_indices给三个输入重新排序

    if require_indices:
        return result, shuffle_indices
    else:
        return result