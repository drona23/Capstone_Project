export function formatPercent(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`
}

export function formatLatency(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(1)} ms`
}

export function formatDatetimeLabel(value: string): string {
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(new Date(value))
}

export function toDatetimeLocalValue(value: string): string {
  const date = new Date(value)
  const year = date.getFullYear()
  const month = `${date.getMonth() + 1}`.padStart(2, '0')
  const day = `${date.getDate()}`.padStart(2, '0')
  const hours = `${date.getHours()}`.padStart(2, '0')
  const minutes = `${date.getMinutes()}`.padStart(2, '0')
  return `${year}-${month}-${day}T${hours}:${minutes}`
}

export function clampIsoDatetime(value: string, minValue: string, maxValue: string): string {
  const current = new Date(value).getTime()
  const min = new Date(minValue).getTime()
  const max = new Date(maxValue).getTime()
  if (Number.isNaN(current)) {
    return toDatetimeLocalValue(minValue)
  }
  if (current < min) {
    return toDatetimeLocalValue(minValue)
  }
  if (current > max) {
    return toDatetimeLocalValue(maxValue)
  }
  return toDatetimeLocalValue(new Date(current).toISOString())
}
