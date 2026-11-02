import tempfile
import unittest
from pathlib import Path
from common import Journal, correct, parse_answer, summary

class ProtocolTests(unittest.TestCase):
    def test_aqua_final_not_first_capital(self):
        self.assertEqual(parse_answer('Because A is too large. The answer is (D).','AQuA'),'D')
        self.assertIsNone(parse_answer('A) 3 B) 4 C) 5 D) 6 E) 7','AQuA'))
        self.assertIsNone(parse_answer('The answer is A. The answer is B.','AQuA'))
        self.assertEqual(parse_answer('Answer: C','AQuA'),'C')
        self.assertIsNone(parse_answer('An apple is red','AQuA'))
    def test_numeric_final_not_last_number(self):
        self.assertEqual(parse_answer('We use 7 - 3 = 4. The answer is 4. Problem 2 comes next.','AddSub'),'4')
        self.assertIsNone(parse_answer('Compute 7 - 3 = 4. Example 9','AddSub'))
        self.assertEqual(parse_answer('The answer in Arabic numerals is 1,234.5.','AddSub'),'1234.5')
        self.assertTrue(correct(parse_answer('Answer: -3/2','AddSub'),'-1.5','AddSub'))
        self.assertEqual(parse_answer('#### 15','AddSub'),'15')
        self.assertEqual(parse_answer('\\boxed{4}','AddSub'),'4')
        self.assertIsNone(parse_answer('Answer: 3. Answer: 4.','AddSub'))
        self.assertIsNone(parse_answer('<think>Answer: 4','AddSub'))
        self.assertTrue(correct('4.0005','4','AddSub'))
        self.assertFalse(correct(None,'4','AddSub'))
    def test_journal_resume_and_truncated_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'result.jsonl'
            j=Journal(p,{'model':'test'})
            j.append({'index':0,'correct':True,'prediction':'A','truncated':False})
            with p.open('ab') as f:
                f.write(b'{"index": 1')
            resumed=Journal(p,{'model':'test'})
            self.assertEqual(len(resumed.rows),1)
            resumed.append({'index':1,'correct':False,'prediction':None,'truncated':True})
            self.assertEqual(len(Journal(p,{'model':'test'}).rows),2)
            self.assertIsNone(summary(resumed.rows.values(),3)['accuracy'])
            self.assertEqual(summary(resumed.rows.values(),2)['accuracy'],50)
            with self.assertRaises(ValueError): Journal(p,{'model':'different'})
            with self.assertRaises(ValueError): resumed.append({'index':1})

if __name__ == '__main__':
    unittest.main()
