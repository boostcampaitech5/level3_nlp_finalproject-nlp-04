"use client";

import {
	Title,
	Text,
	Tab,
	TabList,
	TabGroup,
	TabPanel,
	TabPanels,
	Grid,
	Col,
	Card,
} from "@tremor/react";
import TreemMap from "./elements/TreeMap";
import SummaryCard from "./elements/Summary";

export default function App() {

	return (
		<main className="px-12 py-12 2xl:mx-32">
			<Title>Deguel</Title>
			<Text>현재, 언론사에서 가장 많이 다루고 있는 키워드 입니다. </Text>

			<TabGroup className="mt-6">
				<TabList>
					<Tab>👍 긍정 키워드</Tab>
					<Tab>👎 부정 키워드</Tab>
					{/*<Tab>📰 뉴스 요약본</Tab>*/}
				</TabList>
				<TabPanels>
					<TabPanel>
						<div className="mt-6">
							<TreemMap className="h-80" color="green" title="긍정 키워드"/>
						</div>
					</TabPanel>
					
					<TabPanel>
						<div className="mt-6">
                            <TreemMap className="h-80" color="tomato" title="부정 키워드"/>
						</div>
					</TabPanel>
				</TabPanels>
			</TabGroup>

			<Title className="mt-6">사용자 추천 기사</Title>
			<Text>많은 사용자가 본 뉴스에요. </Text>

			<Grid numItemsLg={2} className="gap-6 mt-6">
				<Col numColSpanLg={1}>
					<Card>
						<SummaryCard isMain={true}>
							<div className="h-96" />
						</SummaryCard>
					</Card>
				</Col>

				<Col numColSpanLg={1}>
					<Card>
						<SummaryCard isMain={true}>
							<div className="h-96" />
						</SummaryCard>
					</Card>
				</Col>
			</Grid>
		</main>
	);
}
