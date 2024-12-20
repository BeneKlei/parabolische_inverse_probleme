

from pathlib import Path
from typing import Dict, Union
from datetime import datetime

from pymor.core.pickle import dump

def save_dict_to_pkl(path: Union[str, Path],
                     data: Dict) -> None:

    path = Path(path)
    assert path.suffix in ['.pkl', 'pickle']
    assert path.parent.exists()

    assert isinstance(data, Dict)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_with_timestamp = f"{timestamp}_{path.stem}{path.suffix}"
    path_with_timestamp = path.parent / filename_with_timestamp
    
    with open(path_with_timestamp, 'wb') as file:
        dump(data, file)