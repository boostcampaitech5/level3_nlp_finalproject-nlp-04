"use client";

import {
	Title,
	Tab,
	TabList,
	TabGroup,
	TabPanel,
	TabPanels,
	Metric,
} from "@tremor/react";

import TreemMap from "./elements/TreeMap";

export default function App() {

	return (
		<main className="px-12 py-12 2xl:mx-32">
			<Metric>Deguel</Metric>
			<Title>현재, 언론사에서 가장 많이 다루고 있는 키워드 입니다. </Title>

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
		</main>
	);
}
