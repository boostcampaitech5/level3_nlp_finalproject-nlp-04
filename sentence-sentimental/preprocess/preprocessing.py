import re
import pandas as pd

def remove_text(texts) :
    pattern = r"\*? ?재판매 및 DB 금지"
    
    preprocessed_text = ''
    
    text = re.sub(pattern, "", texts).strip()
    if text:
        preprocessed_text = text
    
    return preprocessed_text

def remove_press(texts):
    """
    언론 정보를 제거
    -> ~~ 기자 (연합뉴스)
    -> (서울=연합뉴스) ~~ 특파원
    """
    re_patterns = [
        r"\([^(]*?(뉴스|경제|일보|미디어|데일리|한겨례|타임즈|위키트리)\)",
        r"[가-힣]{0,4} (기자|선임기자|수습기자|특파원|객원기자|논설고문|통신원|연구소장) =?",  # 이름 + 기자
        r"[가-힣]{1,}(뉴스|경제|일보|미디어|데일리|한겨례|타임|위키트리)",  # (... 연합뉴스) ..
        r"[\(\[]\s+[\)\]]",  # (  )
        r"[\(\[]=\s+[\)\]]",  # (=  )
        r"[\)\]]\s+=[\)\]]",  # (  =)
    ]

    preprocessed_text = ''
    for re_pattern in re_patterns :
        text = re.sub(re_pattern, "", str(texts))
    if text :
        preprocessed_text = text
            
    return preprocessed_text

def remove_photo_info(texts):
    ## 수정 필요
    """
    뉴스의 이미지 설명 대한 label 제거
    """
    preprocessed_text = []

    preprocessed_text = re.sub(r"\(출처 ?= ?.+\) |\(사진 ?= ?.+\) |\(자료 ?= ?.+\)| \(자료사진\) |사진=.+기자 ", "", texts).strip()
    preprocessed_text = re.sub(r"\/사진제공?=\S+?\s+ | \/\s사진?=", " ", preprocessed_text)
            
    return preprocessed_text

def change_quotation(texts) :
    pattern = r"\""
    replacement = "\'"
    
    processed_text = re.sub(pattern, replacement, texts)
    
    return processed_text

def remove_email(texts):
    """
    이메일을 제거
    """
    preprocessed_text = ''
    
    text = re.sub(r"[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "", texts).strip()
    if text:
        preprocessed_text = text
        
    return preprocessed_text

def remove_day(texts):
    """
    날짜와 관련된 숫자 제거
    """
    pattern = r'\d{4}\.\d{2}\.\d{2}'
    
    text = re.sub(pattern, "", texts)
    return text

def remove_triangle(texts):
    pattern = r'▶?\s.+='
    
    text = re.sub(pattern, "", texts)
    return text

def remove_parentheses(texts):
    """
    괄호와 그 안에 있는 내용들을 제거
    """
    pattern = r'[\(\[][^(^[]*[\)\]]'
    
    processed_text = re.sub(pattern, '', texts)
    
    return processed_text

def remove_copyright(texts):
    """
    뉴스 내 포함된 저작권 관련 텍스트를 제거합니다.
    """
    re_patterns = [
        r"\<저작권자(\(c\)|ⓒ|©|\(Copyright\)|(\(c\))|(\(C\))).+?\>",
        r"저작권자\(c\)|ⓒ|©|(Copyright)|(\(c\))|(\(C\))"
    ]
    preprocessed_text = ''
    
    for re_pattern in re_patterns :
        text = re.sub(re_pattern, "", texts)
    if text:
        preprocessed_text = text
        
    return preprocessed_text

def split_sentence(texts) :
    sentence_list = texts.split(". ")
    
    return sentence_list
    
# 혹시 필요한 것들 모음
def remove_hashtag(texts):
    """
    해쉬태그(#)를 제거합니다.
    """
    preprocessed_text = ''
    
    text = re.sub(r"#\S+", "", texts).strip()
    if text:
        preprocessed_text = text
        
    return preprocessed_text

def remove_url(texts):
    """
    URL을 제거합니다.
    주소: www.naver.com`` -> 주소:
    """
    preprocessed_text = ''

    text = re.sub(r"(http|https)?:\/\/\S+\b|www\.(\w+\.)+\S*", "", texts).strip()
    text = re.sub(r"pic\.(\w+\.)+\S*", "", texts).strip()
    
    preprocessed_text = text
        
    return preprocessed_text

def all_preprocessing(texts) :
    texts = remove_text(texts)
    texts = remove_press(texts)
    texts = remove_photo_info(texts)
    texts = remove_email(texts)
    texts = remove_copyright(texts)
    texts = remove_day(texts)
    texts = remove_triangle(texts)
    texts = remove_parentheses(texts)
    texts = remove_copyright(texts)
    
    texts = change_quotation(texts)
    
    return texts

def preprocess_dataframe(df):
    df['content_corpus'] = df['content'].apply(all_preprocessing)
    
    return df

def preprocess_dataframe_to_sentence(df):
    df_sentence = pd.DataFrame(columns=['title', 'date', 'content_sentence'])

    for index, row in df.iterrows():
        title = row['title']
        date = row['date']
        content = row['content']
        
        content = split_sentence(content)

        for sentence in content:
            l = len(df_sentence)
            sentence = all_preprocessing(sentence)
            new_row = {'title': title, 'date': date, 'content_sentence': sentence}
            
            new_row = pd.DataFrame(new_row, index = [l])
            df_sentence = pd.concat([df_sentence, new_row], axis = 0)

    return df_sentence

if __name__ == "__main__" :
    text = '여기에 넣고 싶은 문구를 넣어서 실험해보세요.'
    print(text)