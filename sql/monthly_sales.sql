
SELECT
    TO_CHAR(o.invoice_date, 'YYYY-MM') AS sales_month,
    ROUND(SUM(oi.revenue), 2) AS monthly_revenue,
    COUNT(DISTINCT o.invoice_no) AS total_orders,
    SUM(oi.quantity) AS units_sold
FROM orders o
JOIN order_items oi
    ON o.invoice_no = oi.invoice_no
GROUP BY TO_CHAR(o.invoice_date, 'YYYY-MM')
ORDER BY sales_month;
