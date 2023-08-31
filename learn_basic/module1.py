def rand_num_kv2(first_name, last_name, **toppings):
    info = {}
    info['first_name'] = first_name
    info['last_name'] = last_name
    for k, v in toppings.items():
        info[k] = v

    return info
