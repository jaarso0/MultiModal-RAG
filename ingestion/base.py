#Anything that calls itself an extractor must have an extract method that takes a file path and returns a list of dictionaries, No exceptions.

#That "pass" means the body is empty - there is no logic, no file reading, nothing. 
#its purely saying "hey, if you want to be an extractor, you have to have this method, and it has to take these arguments and return this type of data".
#"a method that called extract must exist on every class that inherits from me"


#base.py is creates a guarenteed interface


from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseExtractor(ABC): #class inherits from ABC- built in python module for abstract classes
        @abstractmethod
        async def extract(self, file_path: str)->List[Dict[str, Any]]:
                pass
        


