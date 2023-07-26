import {
		Title,
		Text,
		LineChart,
		TabList,
		Tab,
		TabGroup,
		TabPanel,
		TabPanels,
		Flex,
		BadgeDelta,
		Metric,
} from "@tremor/react";

import { useEffect, useState } from "react";
import { startOfYear, subDays } from "date-fns";
import { supabase } from "../supabaseClient";

// 원화 표기를 위한 함수. 
const dataFormatter = (number) => `${Intl.NumberFormat("ko-KR", {
	style: 'currency',
	currency: 'KRW',
}).format(number).toString()}`;

export default function LineChartTab() {
	const [selectedIndex, setSelectedIndex] = useState("Max");
	const [stockPrice, setStockPrice] = useState([{"id": 0, "ticker": "005930.KS", "price": 0, "date": "2021-04-05"}, {"id": 0, "ticker": "005930.KS", "price": 0, "date": "2021-04-05"}]);

	// useEffect로 함수 call. 
	useEffect(() => {
		getInformations();
	}, []);


	// Supabase에서 데이터 가져오기. 
	async function getInformations() {
		const { data } = await supabase.from("price").select("*").order('date', { ascending: true });
		setStockPrice(data);
		getFilteredData(selectedIndex);
	}

	// 날짜 포멧 변경
	const getDate = (dateString) => {
		const [year, month, day] = dateString.split("T")[0].split("-").map(Number);
		return new Date(year, month - 1, day);
	};

	// 주어진 범위의 가격만 가져오기. 
	const filterData = (startDate, endDate) =>
		stockPrice.filter((item) => {
			const currentDate = getDate(item.date);
			return currentDate >= startDate && currentDate <= endDate;
	});

	// 일자 범위에 맞는 만큼의 데이터 가져오기. 
	const getFilteredData = (period) => {
		const lastAvailableDate = getDate(stockPrice[stockPrice.length - 1].date);
		switch (period) {
			case 0: {
				const periodStartDate = subDays(lastAvailableDate, 30);
				return filterData(periodStartDate, lastAvailableDate);
			}
			case 1: {
				const periodStartDate = subDays(lastAvailableDate, 60);
				return filterData(periodStartDate, lastAvailableDate);
			}
			case 2: {
				const periodStartDate = subDays(lastAvailableDate, 180);
				return filterData(periodStartDate, lastAvailableDate);
			}
			case 3: {
				const periodStartDate = startOfYear(lastAvailableDate);
				return filterData(periodStartDate, lastAvailableDate);
			}
			default:
				return stockPrice;
		}
	};

	const getDiffRatio = (price1, price2) => {
		return (price1 - price2) / price2 * 100;
	}

	return (
		<div>
			<Flex>
				<Title>삼성전자</Title>
				<BadgeDelta deltaType="moderateDecrease" isIncreasePositive={true} size="xs">
						{getDiffRatio(stockPrice[stockPrice.length - 1].price, stockPrice[stockPrice.length - 2].price).toFixed(2)}%
				</BadgeDelta>
			</Flex>
			<Metric>{dataFormatter(stockPrice[stockPrice.length - 1].price)}</Metric>
			<Text>최근 1주일간의 주식 차트에요!</Text>
			<TabGroup index={selectedIndex} onIndexChange={setSelectedIndex} className="mt-10">
				<TabList variant="line">
					<Tab>1M</Tab>
					<Tab>2M</Tab>
					<Tab>6M</Tab>
					<Tab>YTD</Tab>
					<Tab>Max</Tab>
				</TabList>
				<TabPanels>
					<TabPanel>
						<LineChart
							className="h-80 mt-8"
							data={getFilteredData(selectedIndex)}
							index="date"
							categories={["price"]}
							colors={["blue"]}
							valueFormatter={dataFormatter}
							showLegend={false}
							yAxisWidth={48}
						/>
					</TabPanel>
					<TabPanel>
						<LineChart
							className="h-80 mt-8"
							data={getFilteredData(selectedIndex)}
							index="date"
							categories={["price"]}
							colors={["blue"]}
							valueFormatter={dataFormatter}
							showLegend={false}
							yAxisWidth={48}
						/>
					</TabPanel>
					<TabPanel>
						<LineChart
							className="h-80 mt-8"
							data={getFilteredData(selectedIndex)}
							index="date"
							categories={["price"]}
							colors={["blue"]}
							valueFormatter={dataFormatter}
							showLegend={false}
							yAxisWidth={48}
						/>
					</TabPanel>
					<TabPanel>
						<LineChart
							className="h-80 mt-8"
							data={getFilteredData(selectedIndex)}
							index="date"
							categories={["price"]}
							colors={["blue"]}
							valueFormatter={dataFormatter}
							showLegend={false}
							yAxisWidth={48}
						/>
					</TabPanel>
					<TabPanel>
						<LineChart
							className="h-80 mt-8"
							data={getFilteredData(selectedIndex)}
							index="date"
							categories={["price"]}
							colors={["blue"]}
							valueFormatter={dataFormatter}
							showLegend={false}
							yAxisWidth={48}
						/>
					</TabPanel>
				</TabPanels>
			</TabGroup>
		</div>
	);
}  
