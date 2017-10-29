MAX_ITERS = 1000
EPSILON = 0.001
DISCOUNT = 1
GOAL = (3, 6)
GRID_SIZE = 7
WIND_TYPE = 1

def main():
    for i in range(3):
        value_iter(i)
        print()


def value_iter(wind_type):
    util_p = initList()
    iters = 0
    delta = 0
    while True:
        iters += 1
        util = util_p
        util_p = initList()
        delta = 0
        for r in range(7):
            for c in range(7):
                util_p[r][c] = calc_util(util, r, c, wind_type)
                new_delta = abs(util_p[r][c] - util[r][c])
                if new_delta > delta:
                    delta = new_delta
        if delta < EPSILON or iters > MAX_ITERS:
            break
    for x in util_p:
        print(x)


def initList():
    return [[0 for i in range(GRID_SIZE)] for i in range(GRID_SIZE)]


def calc_util(util, r, c, wind_type):
    reward = 0 if (r, c) == GOAL else -1
    best_action_val = get_max_action(util, r, c, wind_type)
    return reward + DISCOUNT * best_action_val


def get_max_action(util, r, c, wind_type):
    '''
    wind_type: 0 = no wind, 1 = light wind, 2 = strong wind
    '''
    best_action = float('-inf')
    for r_action in [-1, 0, 1]:
        for c_action in [-1, 0, 1]:
            r_wind = 0
            if wind_type == 1 and c in [3,4,5]:
                r_wind = -1
            elif wind_type == 2 and c in [3,4,5]:
                r_wind = -2

            r_new = r_action + r + r_wind
            c_new = c_action + c

            # check if out of bounds and readjust new state
            r_new = 0 if r_new < 0 else r_new
            c_new = 0 if c_new < 0 else c_new
            r_new = GRID_SIZE - 1 if r_new >= GRID_SIZE else r_new
            c_new = GRID_SIZE - 1 if c_new >= GRID_SIZE else c_new

            curr_action = util[r_new][c_new]
            if curr_action > best_action:
                best_action = curr_action
    return best_action



if __name__ == "__main__":
    main()
