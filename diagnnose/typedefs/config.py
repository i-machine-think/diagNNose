from typing import Any, Dict, Union, Set, Tuple

ArgDict = Dict[str, Any]
ConfigDict = Dict[str, ArgDict]

ArgsDisjunction = Tuple['RequiredArgs', 'RequiredArgs']

RequiredArgs = Set[Union[str, ArgsDisjunction]]
