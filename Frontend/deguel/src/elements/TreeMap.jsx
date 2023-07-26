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

import {
	Title as TremorTitle,
	Subtitle,
	Button,
	Divider,
	Grid,
	Col,
} from "@tremor/react";


import { color } from 'chart.js/helpers';
import { PropTypes } from 'prop-types';
import { useEffect, useRef, useState } from 'react';

import LineChartTab from './Chart';

import { supabase } from '../supabaseClient';

import { ArrowLeftIcon } from "@heroicons/react/outline";
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

export default function TreeMap(props) {
	const chartRef = useRef();
	const [kewords, setKeywords] = useState([]);
	const [isClicked, setIsClicked] = useState(false);
	const [clickedKeyword, setClickedKeyword] = useState("")

	// useEffect로 함수 call. 
    useEffect(() => {
        getInformations();
    }, []);

    // Supabase에서 데이터 가져오기. 
    async function getInformations() {
        const { data } = await supabase.from("keywords").select("*").order('create_time', { ascending: false });
		console.log(data);
        setKeywords(data);
    }

	TreeMap.propTypes = {
		color: PropTypes.string.isRequired,
		title: PropTypes.string.isRequired,
	}

	const config = {
		type: 'treemap',
		data: {
			datasets: [
				{
					tree: kewords,
					key: props.color === "tomato" ? 'neg_cnt' : 'pos_cnt',
					labels: {
						display: true,
						formatter: (context) => context.raw._data.keyword,
						color: ['white', 'whiteSmoke'],
						font: [{size: 36, weight: 'bold'}, {size: 12}],
					},
					backgroundColor(context) {
						if (context.type !== 'data') return 'transparent';
						const { count, pos_cnt, neg_cnt } = context.raw._data;

						if(props.color === "green")
							return pos_cnt / count === 0
								? color('grey').rgbString()
								: color(props.color).alpha(pos_cnt / count).rgbString();

						if(props.color === "tomato")
							return neg_cnt / count === 0
								? color('grey').rgbString()
								: color(props.color).alpha(neg_cnt / count).rgbString();
					},
				},
			],
		},
	};

	const options = {
		plugins: {
			title: {
				display: true,
				text: '분석된 키워드',
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
	const onDataClick = (event) => {
		if(chartRef.current) {
			console.log(event);
			console.log(chartRef);
			const clicked_text = getElementAtEvent(chartRef.current, event)[0].element.options.labels.formatter;

			setClickedKeyword(clicked_text);
			setIsClicked(true);
		}
	}

	// Click event for Close button.
	const onCloseClick = (event) => {
		setIsClicked(false);
	}

	return (
		<div>
			{!isClicked && <div>
				<Subtitle>분석된 {props.title} 예요. </Subtitle>
				<TremorTitle>{props.title}</TremorTitle>
				<Chart ref={chartRef} type="treemap" data={config.data} options={options} onClick={onDataClick}/>
			</div>}

			{isClicked && <div>
				<Grid numItemsLg={6} className="gap-6 mt-6">
					<Col numColSpanLg={4}>
						<LineChartTab ticker="005930">
							<div className="h-96" />
						</LineChartTab>
					</Col>
	
					<Col numColSpanLg={2}>
						<SummaryCard keyword={clickedKeyword} isMain={false}>
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
