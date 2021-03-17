SELECT SUM(order_quantity) as "Units"
FROM ORDERS
WHERE ORDERS.order_datetime = CURRENT_DATE-1;