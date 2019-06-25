import pandas

wFile = open('concatenated.txt', 'w')
input = pandas.read_csv("output.csv")
length = ['name', 'eatType', 'customer rating', 'familyFriendly', 'near', 'food', 'area', 'priceRange']

for i in range(len(input)):
    string = ""
    for s in length:
        try:
            a = str(input.iloc[i][s])
            if s == 'familyFriendly':
                if a == 'yes':
                    a = 'family friendly'
                else:
                    a = 'not friendly'
            if a == 'nan':
                a = ''
        except:
            a = ''
        string += ' ' + a
    wFile.write(string + '\n')
    print(string)
