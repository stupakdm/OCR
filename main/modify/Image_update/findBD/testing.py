
if __name__=="__main":
    enter = input()
    s1 = dict()
    s1['('] = [i for i,x in enumerate(enter) if x =='(']
    s1[')'] = [i for i,x in enumerate(enter) if x ==')']

    enter = [i for i in enter if i=='(' or i==')' ]

    ind_1 = 0
    ind_2 = 0
    i = 0
    l = len(enter)
    while i < l:
        if enter[i] == '(':
            ind_1+=1


    for i,val in enumerate(enter):
        if val == '(':
            ind_1+=1
