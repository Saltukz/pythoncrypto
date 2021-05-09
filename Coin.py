class Coin:
    def __init__(self, 
            data,
            pair):                
        self.data = data
        self.pair = pair
       

    def __repr__(self):
        return repr(self.data,self.pair)

    def __getitem__(self, key):
        return getattr(self,key)    

    @property
    def maCheck(self):
        if self.data[''] < 0:
            return 'red'
        else:
            return 'green'