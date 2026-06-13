# -*- coding: utf-8 -*-

import re
mySent = ' I am happy to join with you today in what will go down in history as the greatest demonstration for freedom in the history of our nation.'
regEx = re.compile('\\W+') #匹配非英文字母和数字
listOfTokens = regEx.split(mySent)
print(listOfTokens)
