import {
	Subtitle,
	Button,
	Divider,
	Grid,
	Col,
	DateRangePicker,
} from "@tremor/react";
import { ArrowLeftIcon } from "@heroicons/react/outline";

import { ko } from "date-fns/locale";

import { PropTypes } from 'prop-types';
import { useEffect, useRef, useState } from 'react';

import {
	Chart as ChartJS,
	CategoryScale,
	LinearScale,
	BarElement,
	Title,
	Tooltip,
	Legend,
} from 'chart.js';
import { TreemapController, TreemapElement } from 'chartjs-chart-treemap';
import { Chart, getElementAtEvent } from 'react-chartjs-2';
import { color } from 'chart.js/helpers';

import { supabase } from '../supabaseClient';

import LineChartTab from './Chart';
import SummaryCard from './Summary';

ChartJS.register(
	CategoryScale,
	LinearScale,
	BarElement,
	Title,
	Tooltip,
	Legend,
	TreemapController,
	TreemapElement
);

function shuffle(array) {
	array.sort(() => Math.random() - 0.5);
}

export default function TreeMap(props) {
	const chartRef = useRef();
	const [kewords, setKeywords] = useState([]);
	const [isClicked, setIsClicked] = useState(false);
	const [clickedKeyword, setClickedKeyword] = useState("");
	const [clickedTicker, setClickedTicker] = useState("");
	const [updatedTime, setUpdatedTime] = useState("");
	const [windowSize, setWindowSize] = useState([
		window.innerWidth,
		window.innerHeight,
	]);
	const [neg_cntsum, setNeg_cntsum] = useState(0);
	const [pos_cntsum, setPos_cntsum] = useState(0);
	// const [dateRange, setDateRange] = useState({
	// 	from: new Date(),
	// 	to: new Date(),
	// });

	// useEffect로 함수 call. 
    useEffect(() => {
        getInformations();

		const handleWindowResize = () => {
			setWindowSize([window.innerWidth, window.innerHeight]);
		};

		window.addEventListener('resize', handleWindowResize);
		return () => {
			window.removeEventListener('resize', handleWindowResize);
		};
    }, [neg_cntsum, pos_cntsum]);

    // Supabase에서 데이터 가져오기. 
    async function getInformations() {
		// let { data } = await supabase.from("keywords").select("*").order('create_time', { ascending: false }).limit(1);
        // data = await supabase.from("keywords").select("*").eq("create_time", data[0].create_time).order('create_time', { ascending: false });
		// data = data.data
		
		const { data } = await supabase.from("keywords").select("*").order('create_time', { ascending: false });
		let neg = 0, pos = 0

		for(let i = 0;i < data.length;i++) {
			if(data[i].pos_cnt >= data[i].neg_cnt) {
				data[i].ratio_pos = (data[i].pos_cnt - data[i].neg_cnt) * data[i].pos_cnt;
				data[i].ratio_neg = 0;
			}
			if(data[i].pos_cnt <= data[i].neg_cnt) {
				data[i].ratio_neg = (data[i].neg_cnt - data[i].pos_cnt) * data[i].neg_cnt;
				data[i].ratio_pos = 0;
			}
			// data[i].ratio_pos = data[i].pos_cnt;
			// data[i].ratio_neg = data[i].neg_cnt;
			neg += data[i].neg_cnt;
			pos += data[i].pos_cnt;
		}
		setNeg_cntsum(neg);
		setPos_cntsum(pos);
		console.log(data);
		console.log("neg_cntsum: " + neg_cntsum);
		console.log("pos_cntsum: " + pos_cntsum);

		shuffle(data);
        setKeywords(data);

		var time_formated = data[0].create_time.replace("T", " ").split("+")[0].split(":")[0]
		time_formated = time_formated.split("-")[0] + "년 " + 
						time_formated.split("-")[1] + "월 " + 
						time_formated.split("-")[2].split(" ")[0] + "일 " + 
						time_formated.split("-")[2].split(" ")[1] + "시"
		setUpdatedTime(time_formated);

		// dateRange.to.setDate(dateRange.to.getDate() + 1);
		// const range = {
		// 	date_start: dateRange.from.toISOString().slice(0, 19).replace('T', ' '),
		// 	date_end: dateRange.to.toISOString().slice(0, 19).replace('T', ' '),
		// };
		// const url = "http://118.67.133.198:30008/";
		// console.log(url);

		// // fetch를 사용하여 POST 요청 보내기
		// await fetch(url, {
		// 	method: "get",
		// 	// mode: "no-cors",
		// })
		// // .then(response => response.json()) // JSON 형태의 응답을 파싱
		// .then(result => {
		// 	// 요청에 대한 처리
		// 	console.log("결과:", result);
		// })
		// .catch(error => {
		// 	console.error("오류 발생:", error);
		// });
    }

	// TreeMap의 prop 설정. 
	TreeMap.propTypes = {
		color: PropTypes.string.isRequired,
		title: PropTypes.string.isRequired,
	}

	// TreeMap의 디자인 설정. 
	const config = {
		type: 'treemap',
		data: {
			datasets: [
				{
					tree: kewords,
					key: props.color === "tomato" ? 'ratio_neg' : 'ratio_pos',
					labels: {
						display: true,
						formatter: (context) => context.raw._data.keyword,
						color: ['white', 'whiteSmoke'],
						font: [{size: 20, weight: 'bold'}, {size: 12}],
					},
					backgroundColor(context) {
						if (context.type !== 'data') return 'transparent';
						const { count, pos_cnt, neg_cnt, ratio_neg, ratio_pos } = context.raw._data;

						if(props.color === "green")
							return pos_cnt / count === 0
								? color('grey').rgbString()
								: color(props.color).alpha((ratio_pos) / pos_cntsum * 30).rgbString();

						if(props.color === "tomato")
							return neg_cnt / count === 0
								? color('grey').rgbString()
								: color(props.color).alpha((ratio_neg) / neg_cntsum * 30).rgbString();
					},
				},
			],
		},
	};

	// TreeMap의 Tooltip 세부 설정. 
	const options = {
		plugins: {
			title: {
				display: true,
				text: '키워드를 클릭 해보세요!',
			},
			legend: {
				display: false,
				
			},
			tooltip: {
				displayColors: false,
				callbacks: {
					title(items) {
						return items[0].raw._data.keyword;
					},
					label(item) {
						const {
							_data: { pos_cnt, neg_cnt },
						} = item.raw;
						return [
							`긍정 사용 빈도: ${pos_cnt} 번`,
							`부정 사용 빈도: ${neg_cnt} 번`,
						];
					},
				},
			},
		},
	};

	// Click event for TreeMap. 
	const onDataClick = async (event) => {
		if(chartRef.current) {
			const clicked_text = getElementAtEvent(chartRef.current, event)[0].element.options.labels.formatter;

			for(let keyword of kewords) {
				if(keyword.keyword === clicked_text) {
					const idx = props.color === "green" ? keyword.summary_id.pos_news[0] : keyword.summary_id.neg_news[0];

					let { data } = await supabase.from("news_summary").select("origin_id").eq("origin_id", idx);
					data = await supabase.from("news").select("company").eq("id", data[0].origin_id);
					data = await supabase.from("ticker").select("ticker").eq("name", data.data[0].company);

					setClickedTicker(data.data[0].ticker);
					break;
				}
			}

			setClickedKeyword(clicked_text);
			setIsClicked(true);
		}
	}

	// Click event for Close button.
	const onCloseClick = (event) => {
		setIsClicked(false);
	}

	// Get the width of TreeMap.
	const getTreeMapWidth = (width) => {
		if(width > 1024) {
			return 200;
		} else if (width > 720) {
			return 800;
		} else {
			return 1000;
		}
	}

	return (
		<div>
			{!isClicked && <div>
				<Subtitle>{updatedTime}에 분석된 {props.title} 예요. </Subtitle>
				<Chart height={getTreeMapWidth(windowSize[0])} ref={chartRef} 
						type="treemap" data={config.data} options={options} 
						onClick={onDataClick}/>
			</div>}

			{isClicked && <div>
				<Grid numItemsLg={7} className="gap-6 mt-6">
					<Col numColSpanLg={5}>
						<LineChartTab ticker={clickedTicker}>
							<div className="h-96" />
						</LineChartTab>
					</Col>
	
					<Col numColSpanLg={2}>
						<SummaryCard keyword={clickedKeyword} isMain={false} color={props.color}>
							<div className="h-96" />
						</SummaryCard>
					</Col>
				</Grid>
				<Divider />

				<Button icon={ArrowLeftIcon} onClick={onCloseClick}>뒤로가기</Button>
			</div>}
		</div>
	);
}
