import multiprocessing as mp

def job(x):
    return x**2

def main(times=20):
    #### map ####
    pool = mp.Pool(processes=2)
    res = pool.map(job, range(int(times)))
    print('res:{}'.format(res))
    #### apply_async ### 
    res = pool.apply_async(job,(4,))
    print('res.get:{}'.format(res.get())) ## 單一個值
    
    ## apply_async 多值
    # 
    multi_res = [pool.apply_async(job, (i,)) for i in range(times)]
    # 從迭代器中取出
    print('pool.apply_async...')
    print([res.get() for res in multi_res])
if __name__ == '__main__':
    main(times=10)