class GetFunctions:

    def check_digit(self, word):
        word = ''.join([w for w in word if w not in ".,':;"])
        count_digit = 0
        for w in word:
            if w.isdigit():
                count_digit += 1
        if len(word) < 4:
            if count_digit == len(word):
                return True
            else:
                return False
        else:
            if count_digit >= len(word) // 2:
                return True
            elif count_digit >= len(word) // 3 and '2' in word and '0' in word and 'a' in word:
                return True
        return False

    def compare(self, block1, block2):
        for i in range(len(block1)):
            if block1[i] != block2[i]:
                return False
        return True

    def text_recognition(self, block, model):
        word = model.detect_text(block)
        return word

    def check_for_letter(self, word):
        for w in word:
            if w.isalpha():
                return word.find(w)
        return -1

    '''
    def look_digits(self, word):
        digit_count = 0
        for w in word:
            if w.isdigit():
                digit_count+=1
        if digit_count > 4:
            return True
        else:
            return False
    '''