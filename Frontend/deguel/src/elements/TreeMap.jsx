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
import { Chart } from 'react-chartjs-2';

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

import { XIcon } from "@heroicons/react/outline";
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

	// useEffect로 함수 call. 
    useEffect(() => {
        getInformations();
    }, []);

    // Supabase에서 데이터 가져오기. 
    async function getInformations() {
        const { data } = await supabase.from("keyword").select("keywords").order('created_at', { ascending: false });
		console.log(data[0].keywords);
        setKeywords(data[0].keywords);
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
					key: 'capacityMW',
					labels: {
						display: true,
						formatter: (context) => context.raw._data.name,
					},
					backgroundColor(context) {
						if (context.type !== 'data') return 'transparent';
						const { dataCoverage } = context.raw._data;
						return dataCoverage === 0
							? color('grey').rgbString()
							: color(props.color).alpha(dataCoverage).rgbString();
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
						return items[0].raw._data.name;
					},
					label(item) {
						const {
							_data: { capacityMW, dataCoverage },
						} = item.raw;
						return [
							`Export capacity: ${capacityMW} MW`,
							`Data Coverage: ${dataCoverage * 100}%`,
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

			setIsClicked(true);
		}
	}

	// Click event for Close button.
	const onCloseClick = (event) => {
		console.log(event);

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
						<LineChartTab>
							<div className="h-96" />
						</LineChartTab>
					</Col>
	
					<Col numColSpanLg={2}>
						<SummaryCard>
							<div className="h-96" />
						</SummaryCard>
					</Col>
				</Grid>
				<Divider />
				<Button icon={XIcon} onClick={onCloseClick}>닫기</Button>
			</div>}
		</div>
	);
}
