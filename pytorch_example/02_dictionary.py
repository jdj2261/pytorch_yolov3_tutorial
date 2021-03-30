dicts = []
dict = {}

dict['Type'] = 0

key, value = 'test1', 0
dict[key] = value

dicts.append(dict)

for key, value in dict.items():
    print(key, value)
