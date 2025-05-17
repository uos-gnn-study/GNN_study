실험을 작성하기에 앞서, 다음을 지켜주세요!

* 코드 스타일\
  Python 코드는 재사용성을 위해 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)에 따라 정리해주세요.
* 실험 설계 문서\
  실험 설계나 정리 내용은 노션 각 주차 페이지에 문서를 만들어 정리해주세요.\
  예: `5주차 스터디/안재현 - 엄준혁 실험 설계`
* 코드 실행 방식\
  가능하면 `bash` 스크립트 없이, Python 코드만으로 동작하도록 설계해주세요.
* 하이퍼파라미터 관리\
  예외적으로 하이퍼파라미터는 `conf.yaml` 등 별도의 yaml 파일로 분리해 관리할 수 있도록 해주세요.
* 깃헙 업로드 위치\
  실험 코드는 `lab/` 디렉토리 하위에 각 실험 폴더를 만들어 업로드해주세요.\
  폴더 이름 예: `GAT_PPI_ex1`
* 필수 파일 첨부\
  실험 폴더에는 반드시 `requirements.txt`를 첨부해주세요.
* README 작성\
  각 실험 폴더 내 `README.md`에 실험 결과와 실행 방법을 정리해주세요.\
  특히 **사용한 하이퍼파라미터 수**는 반드시 명시해주세요.
* 랜덤 시드 명시\
  실험 재현을 위해 `randn`, `shuffle` 등을 사용한 경우, 사용한 **시드(seed)**를 명확히 기록해주세요.
* 커밋 메시지 규칙\
  커밋 메시지 규칙은 다음 가이드를 참고해주세요.\
  [Conventional Commits - qoomon/gist](https://gist.github.com/qoomon/5dfcdf8eec66a051ecd85625518cfd13)