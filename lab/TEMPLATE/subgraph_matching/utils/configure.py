"""`yaml`형 구성 정보를 직관적으로 불러오기 위한 모듈입니다.

`yaml`형 데이터에서 구성 정보를 불러올 때, 타입 힌트가 불충분한 문제가 발생합니다.
따라서 구성 정보를 정적으로 불러올 방법을 제공하는 클래스가 필요합니다.
일부의 제약과 함께 yaml 파일로 저장된 구성 정보를 불러올 방법을 제공합니다.
모든 구성 정보는 속성을 통해 접근할 수 있으며 `ConfigureItem` 타입으로 추론됩니다.

Example:

    conf = Configure(path="./conf/data.yaml")
    conf.path       # data.yaml에서 `path`에 저장된 정보
"""

from typing import TypeAlias, Mapping, Sequence
from os import PathLike
import os.path as osp
import yaml

ConfigureItem: TypeAlias = Sequence["ConfigureItem"] | Mapping[str, "ConfigureItem"] | int | float | str

class Configure:
    """`yaml`로 저장된 구성 정보를 처리하는 클래스입니다.
    
    구성 정보를 보다 직관적으로 처리하기 위해 정의되었습니다.
    인스턴스를 정의할 때 자동으로 데이터를 불러와 속성으로 정의합니다.

    Examples:
    
        conf = Configure(path="./conf/data.yaml")
        conf.path       # data.yaml에서 `path`에 저장된 정보
    """

    __slots__ = ("_store")
    
    def __init__(self, path: PathLike) -> None:
        """`path`에 저장된 구성 정보를 불러옵니다.
        
        `path`에는 `yaml`파일의 주소가 입력될 것으로 기대합니다. 
        그 외의 경우는 `TypeError`를 발생시킵니다.
        또한 구성 정보는 매핑 형식으로 제공될 것으로 기대합니다

        Args:
            path: 구성 정보를 저장한 파일의 주소를 가리키는 문자열

        Raises:
            TypeError: `yaml`이 아닌 파일을 사용하여 발생하는 오류
            ValueError: 구성 정보가 매핑 형식이 아닌 경우 발생하는 오류
        """

        _yamlFileExtension = {'.yaml', '.yml'}

        _, extension = osp.splitext(path)

        # Exception 1. `yaml` 파일 형식인지 확인하기
        if extension not in _yamlFileExtension:
            raise TypeError(f"`yaml`이 아닌 파일을 불러오려고 시도했습니다: {path}")


        with open(path, "r") as f:
            configure: Mapping[str, ConfigureItem] = yaml.full_load(f)

            # Exception 2. 매핑 형식인지 확인하기 
            if not isinstance(configure, Mapping):
                raise ValueError
            
            # _store에 정보 저장하기
            self._store = configure

    def __getattr__(self, name: str) -> ConfigureItem:
        """인스턴스에 저장된 구성 정보를 불러옵니다.
        
        구성 정보를 불러오는 경우에 적절한 타입 힌트를 주기 어려워 제작되었습니다.
        `name`에 명시된 구성 정보가 존재하면 인스턴스에 저장된 정보를 불러옵니다.
        존재하지 않는 경우 `ValueError`를 발생시킵니다.

        Args:
            name: 불러오고자 하는 구성 정보의 명칭

        Raises:
            ValueError: 구성 정보가 존재하지 않아 발생하는 오류
        """
        if name in self._store:
            return self._store[name]
        raise ValueError(f"구성 정보 {name}는 존재하지 않습니다.")
    
    def __dir__(self) -> list[str]:
        return ["data"]