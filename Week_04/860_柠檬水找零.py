class Solution:
    def lemonadeChange(self, bills):
        # 5元和10可以用于找零
        cash_5 = 0
        cash_10 = 0
        for i in bills:
            if i == 5:
                cash_5 += 1
            elif i == 10:
                if cash_5:
                    cash_5 -= 1
                    cash_10 += 1
                else:
                    if not cash_5:
                        return False
            elif i == 20:
                # 先用10 找零
                if cash_10 and cash_5:
                    cash_10 -= 1
                    cash_5 -= 1
                elif cash_5 >= 3:
                    cash_5 -= 3
                else:
                    return False
        return True
