## D-Meta.ai 과제 전형

과제에 대한 솔루션 설명 자료는 [솔루션자료_디메타 과제 전형.pdf](https://github.com/Cafelatte1/MyPortfolio/blob/main/Project/Company/D-Meta/%EA%B3%BC%EC%A0%9C%20%EC%A0%84%ED%98%95/%EC%86%94%EB%A3%A8%EC%85%98%EC%9E%90%EB%A3%8C_%EB%94%94%EB%A9%94%ED%83%80%20%EA%B3%BC%EC%A0%9C%20%EC%A0%84%ED%98%95.pdf) 파일이니 먼저 참고 부탁드립니다.

소스코드는 'sources' 폴더 안에 있습니다.

이미지는 꼭 실행하는 스크립트와 같은 depth 내 'data' 폴더 안에 있어야 합니다.

root
  ㄴcut_image.py
   ㄴ data\
      ㄴ image1.jpg

## 소스 실행 Code Snipet

```
# 1. cut_image.py
# format
python cut_image.py ${input_file_name} ${M} ${N} ${prefix} ${seed}
# example
python .\cut_image.py yerin2.jpg 2 2 test 42

# 2. merge_image.py
# format
python merge_image.py ${input_file_name} ${M} ${N} ${output_file_path} ${seed}
# example
python .\merge_image.py yerin2.jpg 2 2 test_merge.jpg 42

# INFO : image must be in 'data' folder
```







# Project Title

One Paragraph of project description goes here

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

