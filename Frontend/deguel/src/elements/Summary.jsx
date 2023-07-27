import {
    Title,
    Subtitle,
    Text,
    Tracker,
    Divider,
    Flex,
    BadgeDelta,
} from "@tremor/react"

import { PropTypes } from 'prop-types';
import { useEffect, useState } from "react";

import { supabase } from "../supabaseClient";

const data_tracker = [
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
    { color: "gray", tooltip: "언급 없음" },
];

export default function SummaryCard(props) {
    const [newsSummarized, setNewsSummarized] = useState("")
    const [keyword, setKeyword] = useState(props.keyword)
    const [dataTracker, setDataTracker] = useState(data_tracker)
    const [href, setHref] = useState("")

    // useEffect로 함수 call. 
    useEffect(() => {
        getInformations();
    }, []);

    // SummaryCard의 props 설정. 
    SummaryCard.propTypes = {
		keyword: PropTypes.string.isRequired,
        color: PropTypes.string.isRequired,
        isMain: PropTypes.bool.isRequired,
	}

    // start와 end 사이의 날짜 list 반환 함수. 
    const getDateList = (start, end) => {
        for(var arr=[],dt=new Date(start); dt<=new Date(end); dt.setDate(dt.getDate()+1)){
            arr.push(new Date(dt));
        }
        return arr;
    }

    // Supabase에서 데이터 가져오기. 
    async function getInformations() {
        console.log("keyword :" + props.keyword);
        let { data } = await supabase.from("keywords").select("summary_id").eq("keyword", props.keyword);

        for(let d of data) {
            const id = props.color === "green" ? d.summary_id.pos_news[0] : d.summary_id.neg_news[0];

            data = await supabase.from("news_summary").select("summarization").eq("origin_id", id);
            if(data.data[0].summarization !== ""){
                setNewsSummarized(data.data[0].summarization);
                break;
            }
        }

        data = await supabase.from("keywords").select("create_time").eq("keyword", props.keyword);
        
        const list_date = getDateList(new Date() - 30 * 24 * 60 * 60 * 1000, new Date());
        data = data.data.map(item => item.create_time.split("T")[0]);

        const newDataTracker = JSON.parse(JSON.stringify(data_tracker));

        // activation graph 적용. 
        for(const [index, item] of newDataTracker.entries()){
            const year = list_date[index].getFullYear();
            const month = String(list_date[index].getMonth() + 1).padStart(2, '0'); // 월은 0부터 시작하므로 +1을 해주고, 두 자리 숫자로 포맷팅
            const day = String(list_date[index].getDate()).padStart(2, '0'); // 두 자리 숫자로 포맷팅

            // `2023-07-25` 형식으로 변환
            const formattedDate = `${year}-${month}-${day}` + "";
            newDataTracker[index].tooltip = formattedDate;

            if(data.includes(formattedDate)) {
                newDataTracker[index].color = "green";
            } else {
                newDataTracker[index].color = "gray";
            }
        }
        data = await supabase.from("news").select("link").eq("id", id);
        setHref(data.data[0].link);

        setDataTracker(newDataTracker);
    }

    // Click event for Close button.
	const onCloseClick = (text) => {
        console.log("+++ badge +++")
        setKeyword(text);
	}

    return (
        <div>
            {!props.isMain && <Subtitle>선택하신 키워드에 대한 주요 뉴스 요약본이에요!</Subtitle>}
            <Title># {keyword}</Title>
            <a href={href} target="_blank" rel="noreferrer"><Text color="black">{newsSummarized}</Text></a>
            {!props.isMain &&
                <div>
                    <Divider />
                    <Subtitle>해당 키워드의 지난 30일간 언급 유무예요. </Subtitle>
                    <Title>언급 빈도</Title>
                    <Tracker data={dataTracker} className="mt-2" />
                    <Divider />

                    <Subtitle>같이 언급된 키워드예요. </Subtitle>
                    <Title>연관 키워드</Title>
                    <Flex justifyContent="start" className="overflow-scroll gap-2 mt-2">
                        <BadgeDelta size="md" deltaType="increase" onClick={() => onCloseClick("양산화")}>양산화</BadgeDelta>
                        <BadgeDelta size="md" deltaType="increase" onClick={() => onCloseClick("평택 공장")}>평택 공장</BadgeDelta>
                        <BadgeDelta size="md" deltaType="increase" onClick={() => onCloseClick("증설")}>증설</BadgeDelta>
                        <BadgeDelta size="md" deltaType="decrease" onClick={() => onCloseClick("조사")}>조사</BadgeDelta>
                        <BadgeDelta size="md" deltaType="decrease" onClick={() => onCloseClick("TSMC")}>TSMC</BadgeDelta>
                    </Flex>
                </div>
            }
        </div>
    );
}
