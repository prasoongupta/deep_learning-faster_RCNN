# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 13:17:13 2018

@author: prasoon
"""

import unittest

from unittest.mock import MagicMock


def fun(x):
    return x + 1

class MyTest(unittest.TestCase):
    def setUp():
        pass
    def test_(self):
        self.assertEqual(fun(3),3)
    def tearDown():
        pass
    
if __name__=='__main__':
    unittest.main()        