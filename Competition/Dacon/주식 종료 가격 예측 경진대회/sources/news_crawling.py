# ===== 뉴스 기사 크롤링 =====
def getSentimentalScore(sector_dic, from_date, to_date, pages):
    sector_dic_copy = copy.deepcopy(sector_dic)
    cnt = 0
    try:
        # 기본 검색창을 셋팅
        driver = webdriver.Edge('./msedgedriver.exe')
        baseUrl = "https://www.google.com/search?q=test"
        driver.get(baseUrl)

        # okt tokeninzer 초기화
        okt = Okt()
        # 한글 및 1칸의 공백을 추출하기 위한 정규식 기입
        korean_exp = re.compile('[^ ㄱ-ㅣ가-힣]')
        # 불용어 데이터 로드
        stopwords = pd.read_csv("https://raw.githubusercontent.com/yoonkt200/FastCampusDataset/master/korean_stopwords.txt").values.tolist()

        first_flag = True
        for j in list(sector_dic_copy.keys()):
            print("===== get sentimental score on", j, "=====")

            query_word = j + " 주가 전망"
            sector_dic_copy[j]["news_sent"] = []
            if first_flag:
                # 도구를 클릭합니다
                driver_obj = driver.find_element_by_css_selector("#hdtb-tls")
                driver_obj.click()
                sleep(0.5)
                # 모든 날짜를 클릭합니다
                driver_obj = driver.find_element_by_css_selector("#hdtbMenus > span:nth-child(3) > g-popup > div.rIbAWc > div")
                driver_obj.click()
                sleep(0.5)
                # 기간 설정을 클릭합니다
                driver_obj = driver.find_element_by_css_selector("#lb > div > g-menu > g-menu-item:nth-child(8) > div > div")
                driver_obj.click()
                # 찾을 시작할 날짜를 지정합니다
                # from date
                driver_obj = driver.find_element_by_css_selector("#OouJcb")
                driver_obj.click()
                driver_obj.send_keys(from_date)
                # 찾을 마지막을 날짜를 지정합니다
                # to date
                driver_obj = driver.find_element_by_css_selector("#rzG2be")
                driver_obj.click()
                driver_obj.send_keys(to_date)
                # 기간 입력 후 실행을 클릭합니다
                driver_obj = driver.find_element_by_css_selector("#T3kYXe > g-button")
                driver_obj.click()
                sleep(0.5)
                # 검색어를 입력하여 검색합니다
                driver_obj = driver.find_element_by_css_selector("#tsf > div:nth-child(1) > div.A8SBwf > div.RNNXgb > div > div.a4bIc > input")
                driver_obj.clear()
                # driver_obj = driver.find_element_by_css_selector("#tsf > div:nth-child(1) > div.A8SBwf > div.RNNXgb > div > div.a4bIc > input")
                driver_obj.send_keys(query_word)
                driver_obj = driver.find_element_by_css_selector("#tsf > div:nth-child(1) > div.A8SBwf > div.RNNXgb > button > div > span > svg")
                driver_obj.click()
                sleep(0.5)
                # 뉴스를 클릭합니다
                driver_obj = driver.find_elements_by_css_selector("#hdtb-msb > div:nth-child(1) > div > div:nth-child(n) > a")
                for z in driver_obj:
                    if z.text == "뉴스":
                        z.click()
                        break
                sleep(0.5)
                first_flag = False
            else:
                # 크롤링 방지 시스템을 랜덤성 이벤트로 무력화
                cnt += rnd.randint(3)
                if cnt == 10:
                    if rnd.random() > 0.5:
                        driver_obj = driver.find_element_by_css_selector("#hdtbMenus")
                        driver_obj.click()
                        sleep(1 + rnd.randint(3))
                        cnt = 0
                    else:
                        driver_obj = driver.find_element_by_css_selector("#rcnt")
                        driver_obj.click()
                        sleep(1 + rnd.randint(3))
                        cnt = 0
                # 검색어를 입력하여 검색합니다
                driver_obj = driver.find_element_by_css_selector("#lst-ib")
                driver_obj.clear()
                sleep(0.5 + rnd.random())
                # driver_obj = driver.find_element_by_css_selector("#tsf > div:nth-child(1) > div.A8SBwf > div.RNNXgb > div > div.a4bIc > input")
                driver_obj.send_keys(query_word)
                driver_obj = driver.find_element_by_css_selector("#mKlEF > span > svg")
                driver_obj.click()
                sleep(0.5)

            # 크롤링 할 페이지 수에 대한 loop
            for _ in range(pages):
                # 각 기사 title 에 대한 css 참조 샘플
                # rso > div:nth-child(1) > g-card > div > div > a > div > div.iRPxbe > div.mCBkyc.JQe2Ld.nDgy9d
                # rso > div:nth-child(2) > g-card > div > div > a > div > div.iRPxbe > div.mCBkyc.JQe2Ld.nDgy9d

                # 기사가 있는 body 부분을 선택
                driver_obj = driver.find_elements_by_css_selector("#rso > div:nth-child(n) > g-card > div > div > a > div > div.iRPxbe > div.mCBkyc.JQe2Ld.nDgy9d")
                # 만약 기사가 없으면 loop 탈출
                if len(driver_obj) == 0:
                    break

                # 기사 수 만큼 제목을 추출한 후 기업 딕셔너리에 리스트로 저장합니다
                for z in driver_obj:
                    sector_dic_copy[j]["news_sent"].append(z.text)
                # 다음 페이지 설정
                driver_obj = driver.find_elements_by_css_selector('#pnnext')
                # 다음 페이지가 없으면 크롤링 종료
                if len(driver_obj) == 0:
                    break
                else:
                    # 다음 페이지 클릭
                    driver_obj[0].click()
                    if rnd.random() > 0.8:
                        sleep(0.5 + rnd.random())
                    else:
                        sleep(1.5 + rnd.random() / 2)
            print("web crawling complete")
            # z 는 해당 기업 관련된 뉴스기사 텍스트 리스트
            compound_sent = 0
            for z in range(len(sector_dic_copy[j]["news_sent"])):
                # 1. 한글 추출
                sector_dic_copy[j]["news_sent"][z] = korean_exp.sub("", sector_dic_copy[j]["news_sent"][z])
                # 2. okt 모듈을 이용한 tokenizing
                sector_dic_copy[j]["news_sent"][z] = okt.nouns(sector_dic_copy[j]["news_sent"][z])
                # 3. 한 단어로만 이루어진 token 제거 (무의미)
                sector_dic_copy[j]["news_sent"][z] = [i for i in sector_dic_copy[j]["news_sent"][z] if len(i) >= 2]
                # 4. 불용어 제거
                tmp_nouns = []
                # k 는 한 단어
                for k in sector_dic_copy[j]["news_sent"][z]:
                    if [k] not in stopwords: tmp_nouns.append(k)
                sector_dic_copy[j]["news_sent"][z] = tmp_nouns
                # 5. 감성어휘사전을 통한 긍정 및 부정 레이블링
                # -2 :매우 부정, -1: 부정, 0: 중립 or Unknown, 1: 긍정, 2: 매우 긍정
                tmp_sntScore = 0
                for k in sector_dic_copy[j]["news_sent"][z]:
                    sntScore = KnuSL.data_list(k)[1]
                    if sntScore != "None":
                        tmp_sntScore += int(sntScore)
                sector_dic_copy[j]["news_sent"][z] = (sector_dic_copy[j]["news_sent"][z], tmp_sntScore)
                compound_sent += tmp_sntScore
            sector_dic_copy[j]["sent_score"] = compound_sent
            print("text preprocessing complete")
        driver.close()
        return sector_dic_copy
    except:
        print("ERROR : return original")
        return sector_dic

# news_dic 구조
# 1 depth : 기업명
# 2 depth : news_sent(뉴스 기사를 말뭉치로 tokeninzing), sent_score(말뭉치에 대한 긍부정 점수를 합한 값)
news_dic = dict.fromkeys(stock_list["종목명"][:2])
for i in list(news_dic.keys()):
    news_dic[i] = {}

# weekly_news_list
# 검색한 마지막 일자 및 이에 대한 news_dic 을 저장한 튜플
# ex. [ (news_dic, "10/03/2010"), ... ]
weekly_news_list = []
for i in list(range(0,train_x.shape[0]))[:-5]:
    date_to_str = train_x["Date"].dt.strftime('%m/%d/%Y')[(i):(i + 5)]
    print("Date range :", date_to_str.iloc[0], "~", date_to_str.iloc[-1])
    news_dic = getSentimentalScore(news_dic, date_to_str.iloc[0], date_to_str.iloc[-1], 1)
    weekly_news_list.append((news_dic, date_to_str.iloc[-1]))

len(weekly_news_list)

print(train_x.isna().sum())