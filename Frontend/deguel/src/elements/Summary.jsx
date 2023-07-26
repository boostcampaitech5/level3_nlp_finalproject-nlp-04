import { useEffect, useState } from "react";
import {
    Title,
    Subtitle,
    Text,
    Tracker,
    Divider,
    Flex,
    BadgeDelta,
    Button,
} from "@tremor/react"
import { supabase } from "../supabaseClient";

import { PropTypes } from 'prop-types';

const data = [
    { color: "gray", tooltip: "Operational" },
    { color: "gray", tooltip: "Operational" },
    { color: "gray", tooltip: "Operational" },
    { color: "blue", tooltip: "Operational" },
    { color: "gray", tooltip: "Operational" },
    { color: "gray", tooltip: "Operational" },
    { color: "gray", tooltip: "Operational" },
    { color: "gray", tooltip: "Operational" },
    { color: "gray", tooltip: "Operational" },
    { color: "gray", tooltip: "Operational" },
    { color: "gray", tooltip: "Operational" },
    { color: "gray", tooltip: "Operational" },
    { color: "gray", tooltip: "Operational" },
    { color: "gray", tooltip: "Operational" },
    { color: "gray", tooltip: "Operational" },
    { color: "gray", tooltip: "Operational" },
    { color: "blue", tooltip: "Operational" },
    { color: "green", tooltip: "Operational" },
    { color: "green", tooltip: "Operational" },
    { color: "green", tooltip: "Downtime" },
    { color: "blue", tooltip: "Operational" },
    { color: "gray", tooltip: "Operational" },
    { color: "gray", tooltip: "Operational" },
    { color: "gray", tooltip: "Operational" },
    { color: "gray", tooltip: "Maintenance" },
    { color: "gray", tooltip: "Operational" },
    { color: "gray", tooltip: "Operational" },
    { color: "blue", tooltip: "Operational" },
    { color: "green", tooltip: "Degraded" },
    { color: "green", tooltip: "Operational" },
];

export default function SummaryCard(props) {
    const [newsSummarized, setNewsSummarized] = useState("")
    const [keyword, setKeyword] = useState(props.keyword)

    // useEffect로
    // const [newsKeywordList, setnewsKeywordList] = useState([])

    // useEffect로 함수 call. 
    useEffect(() => {
        getInformations();
    }, []);

    SummaryCard.propTypes = {
		keyword: PropTypes.string.isRequired,
        isMain: PropTypes.bool.isRequired,
	}


    // Supabase에서 데이터 가져오기. 
    async function getInformations() {
        const { data } = await supabase.from("news").select("content").order('date', { ascending: false }).limit(1);
        setNewsSummarized(data[0].content);
    }

    // Click event for Close button.
	const onCloseClick = (text) => {
        console.log("+++ badge +++")
        setKeyword(text);
	}

    return (
        <div>
            <Subtitle>선택하신 키워드에 대한 주요 뉴스 요약본이에요!</Subtitle>
            <Title># {keyword}</Title>
            <Text>{newsSummarized}</Text>
            <Divider />
            
            {!props.isMain &&
                <div>
                <Subtitle>해당 키워드의 지난 30일간 언급 유무예요. </Subtitle>
                <Title>언급 빈도</Title>
                <Tracker data={data} className="mt-2" />
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
                    <Divider />
                </div>
            }

            <Subtitle>이 기사를 추천하시나요?</Subtitle>
            <Title>추천하기</Title>
            <div className="flex gap-2 mt-2">
                <div className="flex-1">
                    <Button color="red" className="w-full">🙅‍♂️ 아니요!</Button>
                </div>
                <div className="flex-1">
                    <Button color="green" className="w-full">🙆‍♂️ 네!</Button>
                </div>
            </div>
        </div>
    );
}
