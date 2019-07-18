from typing import Any, Dict, Set

ArgDict = Dict[str, Any]
ConfigDict = Dict[str, ArgDict]

RequiredArgs = Set[str]

ArgDescriptions = Dict[str, Dict[str, Dict[str, Any]]]
