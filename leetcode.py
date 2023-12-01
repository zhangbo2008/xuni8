class Solution: # 压入索引. 弹出时候看索引差.
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        #")()())"
        tmp=[]
        cnt=0
        save=0
        for dex,i in enumerate(s):
            if i=='(':
             tmp.append('(')
            if i==')' and len(tmp)>0 and tmp[-1]=='(':
              cnt+=1
              save=max(save,cnt)
              tmp.pop()
              continue
            if i==')' and len(tmp)==0:
              cnt=0
              tmp.append(')')
              continue
            if i==')'         and len(tmp)>0 and tmp[-1]!='(':  
               cnt=0
               tmp.append(')')
               continue
        return save*2
Solution().longestValidParentheses('()(()')
        
        
            
            
            