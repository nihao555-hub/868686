# -*- coding: utf-8 -*-

import re
text = open("./1.txt").read()
regEx = re.compile("\\W*")
listOfTokens = regEx.split(text)
print([tok.lower() for tok in listOfTokens if len(tok) > 2 ])
