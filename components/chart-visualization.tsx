"use client"
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"

interface ChartVisualizationProps {
  chart: any
  chartType?: string
}

const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884D8", "#82CA9D"]

export function ChartVisualization({ chart, chartType }: ChartVisualizationProps) {
  if (!chart || !chart.series || chart.series.length === 0) {
    return null
  }

  const renderChart = () => {
    // 1) Support categorical series with xAxis.categories and numeric arrays per series
    const hasCategoricalSeries = Array.isArray(chart?.xAxis?.categories)
      && Array.isArray(chart?.series)
      && chart.series.length > 0
      && Array.isArray(chart.series[0]?.data)
      && (typeof chart.series[0].data[0] === "number" || chart.series[0].data.length === 0)

    if (hasCategoricalSeries) {
      const categories: string[] = chart.xAxis.categories
      const seriesDefs: any[] = chart.series

      const transformedData = categories.map((cat: string, idx: number) => {
        const row: Record<string, number | string> = { name: cat }
        seriesDefs.forEach((s: any, sIdx: number) => {
          const key = s.name || `Series ${sIdx + 1}`
          const value = Array.isArray(s.data) && typeof s.data[idx] === "number" ? s.data[idx] : 0
          row[key] = value
        })
        return row
      })

      if (chart.type === "line") {
        return (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={transformedData} margin={{ top: 10, right: 20, left: 10, bottom: 30 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
              <YAxis allowDecimals={false} domain={[0, 'auto']} />
              <Tooltip />
              <Legend verticalAlign="bottom" align="center" wrapperStyle={{ paddingTop: 8 }} />
              {seriesDefs.map((s: any, index: number) => (
                <Line
                  key={index}
                  type="monotone"
                  dataKey={s.name || `Series ${index + 1}`}
                  stroke={s.color || COLORS[index % COLORS.length]}
                  strokeWidth={2}
                  name={s.name || `Series ${index + 1}`}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )
      }

      // default to stacked bars for categorical (like the provided reference chart)
      return (
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={transformedData} margin={{ top: 10, right: 20, left: 10, bottom: 30 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
            <YAxis allowDecimals={false} domain={[0, 'auto']} />
            <Tooltip />
            <Legend verticalAlign="bottom" align="center" wrapperStyle={{ paddingTop: 8 }} />
            {seriesDefs.map((s: any, index: number) => (
              <Bar
                key={index}
                dataKey={s.name || `Series ${index + 1}`}
                fill={s.color || COLORS[index % COLORS.length]}
                name={s.name || `Series ${index + 1}`}
                stackId="stack-0"
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      )
    }

    // 2) Point-based series (array of {x,y}), possibly multiple series
    const isPointSeries = Array.isArray(chart?.series) && chart.series.length > 0 &&
      Array.isArray(chart.series[0]?.data) && typeof chart.series[0].data[0] === "object"

    if (isPointSeries) {
      const seriesDefs: any[] = chart.series
      const xToRow: Record<string | number, any> = {}

      seriesDefs.forEach((s: any, sIdx: number) => {
        const key = s.name || `Series ${sIdx + 1}`
        ;(s.data || []).forEach((pt: any, i: number) => {
          const xVal = pt?.x ?? pt?.name ?? i
          const yVal = pt?.y ?? pt?.value ?? 0
          const row = xToRow[xVal] || { name: xVal }
          row[key] = typeof yVal === "number" ? yVal : Number(yVal) || 0
          xToRow[xVal] = row
        })
      })

      const mergedData = Object.values(xToRow).sort((a: any, b: any) => {
        const ax = Number(a.name)
        const bx = Number(b.name)
        if (!Number.isNaN(ax) && !Number.isNaN(bx)) return ax - bx
        return String(a.name).localeCompare(String(b.name))
      })

      if (chart.type === "line") {
        return (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={mergedData} margin={{ top: 10, right: 20, left: 10, bottom: 30 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
              <YAxis allowDecimals={false} domain={[0, 'auto']} />
              <Tooltip />
              <Legend verticalAlign="bottom" align="center" wrapperStyle={{ paddingTop: 8 }} />
              {seriesDefs.map((s: any, index: number) => (
                <Line
                  key={index}
                  type="monotone"
                  dataKey={s.name || `Series ${index + 1}`}
                  stroke={s.color || COLORS[index % COLORS.length]}
                  strokeWidth={2}
                  name={s.name || `Series ${index + 1}`}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )
      }

      if (chart.type === "bar") {
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={mergedData} margin={{ top: 10, right: 20, left: 10, bottom: 30 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
              <YAxis allowDecimals={false} domain={[0, 'auto']} />
              <Tooltip />
              <Legend verticalAlign="bottom" align="center" wrapperStyle={{ paddingTop: 8 }} />
              {seriesDefs.map((s: any, index: number) => (
                <Bar
                  key={index}
                  dataKey={s.name || `Series ${index + 1}`}
                  fill={s.color || COLORS[index % COLORS.length]}
                  name={s.name || `Series ${index + 1}`}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        )
      }
    }

    // 3) Fallback single-series points
    const chartData = chart.series[0]?.data || []
    const transformedData = chartData.map((item: any, index: number) => ({
      name: item.x || item.name || `Item ${index + 1}`,
      value: item.y || item.value || 0,
      ...item,
    }))

    switch (chart.type) {
      case "line":
        return (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={transformedData} margin={{ top: 10, right: 20, left: 10, bottom: 30 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
              <YAxis allowDecimals={false} domain={[0, 'auto']} />
              <Tooltip />
              <Legend verticalAlign="bottom" align="center" wrapperStyle={{ paddingTop: 8 }} />
              <Line
                type="monotone"
                dataKey="value"
                stroke={COLORS[0]}
                strokeWidth={2}
                name={chart.series[0]?.name || "Series 1"}
              />
            </LineChart>
          </ResponsiveContainer>
        )

      case "bar":
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={transformedData} margin={{ top: 10, right: 20, left: 10, bottom: 30 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
              <YAxis allowDecimals={false} domain={[0, 'auto']} />
              <Tooltip />
              <Legend />
              <Bar
                dataKey="value"
                fill={COLORS[0]}
                name={chart.series[0]?.name || "Series 1"}
              />
            </BarChart>
          </ResponsiveContainer>
        )

      case "pie":
        // Compute total to derive percentage without relying on non-typed `percent` prop
        const total = (transformedData || []).reduce(
          (sum: number, d: any) => sum + (Number(d.value) || 0),
          0
        )
        return (
          <ResponsiveContainer width="100%" height={400}>
            <PieChart>
              <Pie
                data={transformedData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }: any) => {
                  const pct = total > 0 ? ((Number(value) || 0) / total) * 100 : 0
                  return `${name} ${pct.toFixed(0)}%`
                }}
                outerRadius={120}
                fill="#8884d8"
                dataKey="value"
              >
                {transformedData.map((entry: any, index: number) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        )

      default:
        return (
          <div className="flex items-center justify-center h-40 text-muted-foreground">
            Unsupported chart type: {chart.type}
          </div>
        )
    }
  }

  return (
    <div className="space-y-3">
      <h3 className="font-medium">{chart.title || "Data Visualization"}</h3>
      {renderChart()}
    </div>
  )
}