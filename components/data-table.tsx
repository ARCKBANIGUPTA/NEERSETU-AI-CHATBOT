"use client"

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ScrollArea } from "@/components/ui/scroll-area"

interface DataTableProps {
  data: any[]
}

export function DataTable({ data }: DataTableProps) {
  if (!data || data.length === 0) {
    return <div className="text-center py-4 text-muted-foreground">No data available</div>
  }

  const columns = Object.keys(data[0])
  const displayData = data.slice(0, 10) // Limit to first 10 rows for display

  return (
    <div className="space-y-2">
      <ScrollArea className="h-64 w-full rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              {columns.map((column) => (
                <TableHead key={column} className="font-medium">
                  {column.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {displayData.map((row, index) => (
              <TableRow key={index}>
                {columns.map((column) => (
                  <TableCell key={column}>
                    {typeof row[column] === "number" ? row[column].toLocaleString() : row[column]?.toString() || "-"}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </ScrollArea>
      {data.length > 10 && <p className="text-xs text-muted-foreground">Showing first 10 of {data.length} records</p>}
    </div>
  )
}
